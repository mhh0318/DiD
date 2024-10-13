import sys

sys.path.append("/mnt/CV_teamz/users/h1t/code/DiD")

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from noise_schedule_subs import get_noise
import transformers
import models


def get_diffusion_betas(type, T, beta_start, beta_end):
    """Get betas from the hyperparameters."""
    if type == "linear":
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(beta_start, beta_end, T)
    elif type == "cosine":
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = np.arange(T + 1, dtype=np.float64) / T
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = torch.from_numpy(np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999))
        return betas
    elif type == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1.0 / torch.linspace(T, 1.0, T)
    else:
        raise NotImplementedError(type)


class CategoricalDiffusion(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super(CategoricalDiffusion, self).__init__()

        self.config = config

        self.did = self.config.do_did

        # define base model
        if self.config.diffusion_category.backbone == "dit":
            self.backbone = models.dit.DIT(self.config)
        elif self.config.diffusion_category.backbone == "dimamba":
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif self.config.diffusion_category.backbone == "ar":
            self.backbone = models.autoregressive.AR(
                self.config, vocab_size=self.vocab_size, mask_index=self.mask_index
            )
        elif self.config.diffusion_category.backbone == "hf_dit":
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                self.config.eval.checkpoint_path, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.parameterization = self.config.diffusion_category.parameterization
        self.subs_masking = self.config.diffusion_category.subs_masking
        self.patch_size = self.config.diffusion_category.patch_size
        self.softplus = torch.nn.Softplus()
        self.antithetic_sampling = self.config.diffusion_category.antithetic_sampling
        self.sampling_eps = self.config.diffusion_category.sampling_eps
        self.change_of_variables = self.config.diffusion_category.change_of_variables
        self.time_conditioning = self.config.diffusion_category.time_conditioning

        if self.config.diffusion_category.discrete_vae:
            self.vocab_size = self.config.diffusion_category.vocab_size
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.vocab_size = -1
            self.mask_index = -1

        self.T = config.T

        self.dtype = torch.float32 if self.config.dtype == "fp32" else torch.float16

        # define betas and transition matrix
        if config.type_base == "d3pm":  # Original D3PM version. Experimental yet.
            self.diffusion_type = "d3pm"
            betas = get_diffusion_betas(
                self.config.beta_schedule,
                self.T,
                self.config.beta_start,
                self.config.beta_end,
            )
            self.register("betas", betas)

            if self.transition_mat_type == "uniform":
                q_one_step_mats = [
                    self._get_transition_mat(t) for t in range(0, self.num_timesteps)
                ]
            elif self.transition_mat_type == "gaussian":
                q_one_step_mats = [
                    self._get_gaussian_transition_mat(t)
                    for t in range(0, self.num_timesteps)
                ]
            elif self.transition_mat_type == "absorbing":
                q_one_step_mats = [
                    self._get_absorbing_transition_mat(t)
                    for t in range(0, self.num_timesteps)
                ]
            else:
                raise ValueError(
                    f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                    f", but is {self.transition_mat_type}"
                )
            self.register("q_onestep_mats", torch.stack(q_one_step_mats, dim=0))
            assert self.q_onestep_mats.shape == (
                self.num_timesteps,
                self.num_pixel_vals,
                self.num_pixel_vals,
            ), "q_onestep_mats has wrong shape"

            q_mat_t = self.q_onestep_mats[0]
            q_mats = [q_mat_t]
            for t in range(1, self.num_timesteps):
                # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
                q_mat_t = torch.tensordot(
                    q_mat_t, self.q_onestep_mats[t], dims=[[1], [0]]
                )
                q_mats.append(q_mat_t)
            # self.q_mats = torch.stack(q_mats, dim=0)
            self.register("q_mats", torch.stack(q_mats, dim=0))
            assert self.q_mats.shape == (
                self.num_timesteps,
                self.num_pixel_vals,
                self.num_pixel_vals,
            ), self.q_mats.shape

            self.register(
                "transpose_q_onestep_mats", torch.transpose(self.q_onestep_mats, 1, 2)
            )

        elif config.type_base == "subs":  # SUBS type with D3PM absorbing
            self.subs_masking = self.config.diffusion_category.subs_masking
            self.neg_infinity = -1e6
            self.noise = get_noise(self.config, dtype=self.dtype)

        else:
            raise NotImplementedError(f"Not implement {config.parameterizstion}")

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _d3pm_parameterization(self, logits):
        if self.subs_masking:
            logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def _sedd_parameterization(self, logits, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps

        return t

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def mask_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape[:2], device=x.device) < move_chance
        return move_indices

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape[:2], device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def forward(self, latent, label=None, attention_mask=None):
        """Returns log score.
        latent: b c h w, image after VAE
        label: b l, label after embedding
        """
        x0 = self.patchify(latent)

        t = self._sample_t(x0.shape[0], x0.device)

        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        if self.did:
            mask_xt = self.mask_xt(x0, move_chance)

            # sigma = self._process_sigma(unet_conditioning)

            out = self.backbone(x0, mask_xt, unet_conditioning)  # need to add label

            if self.T > 0:
                # this part should be both KL for Cat and KL for Gaussian
                raise NotImplementedError("DID not implemented for T > 0")
            if self.config.loss_type == "mix":
                # discrete+ continuous
                log_p_theta = torch.gather(
                    input=out, dim=-1, index=x0[:, :, None]
                ).squeeze(-1)
            elif self.config.loss_type == "discrete":
                # cluster and 1-hot
                pass
            elif self.config.loss_type == "continuous":
                loss = self.diff_loss(out, x0)
                return loss
        else:
            xt = self.q_xt(x0, move_chance)

            sigma = self._process_sigma(sigma)

            with torch.amp.autocast(dtype=torch.float32):
                out = self.backbone(xt, unet_conditioning)  # need to add label

            logits = out
            if self.parameterization == "subs":
                logits = self._subs_parameterization(logits=logits, xt=xt)
            elif self.parameterization == "d3pm":
                logits = self._d3pm_parameterization(logits=logits)
            elif self.parameterization == "sedd":
                logits = self._sedd_parameterization(logits=logits, xt=xt, sigma=sigma)

            if self.parameterization == "sedd":
                return dsigma[:, None] * self._score_entropy(
                    logits, sigma[:, None], xt, x0
                )

            if self.T > 0:
                diffusion_loss = self._d3pm_loss(model_output=logits, xt=xt, x0=x0, t=t)
                if self.parameterization == "d3pm":
                    reconstruction_loss = self._reconstruction_loss(x0)
                elif self.parameterization == "subs":
                    reconstruction_loss = 0
                return reconstruction_loss + diffusion_loss

            # SUBS parameterization, continuous time.
            log_p_theta = torch.gather(
                input=logits, dim=-1, index=x0[:, :, None]
            ).squeeze(-1)

            if self.change_of_variables or self.importance_sampling:
                return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

            return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    #### =============below are the experimental codes. Disgard all there. =============

    # def forward_d3pm(self, batch, cond):
    #     # experimental !
    #     # a lot of things TODO!
    #     # now only support discrete timespace.
    #     # TODO: try to support continuous timespace.

    #     imgs = batch['image']
    #     x = self.patchify(imgs)
    #     gt_latents = x.clone().detach()

    #     t = torch.randint(1, self.T, (x.shape[0],), device=x.device)

    #     xt = self.q_xt(x, t)

    #     model_output = self.forward(xt, t)
    #     loss = self.diffloss(model_output, gt_latents)
    #     return loss

    # def _at(self, a, t, x):
    #     # t is 1-d, x is integer value of 0 to num_classes - 1
    #     bs = t.shape[0]
    #     t = t.reshape((bs, *[1] * (x.dim() - 1)))
    #     # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
    #     return a[t - 1, x, :]

    # def q_xt(self, x0, t):
    #     logits = torch.log(self._at(self.q_mats, t, x0) + self.eps)
    #     noise = torch.clip(noise, self.eps, 1.0)
    #     gumbel_noise = -torch.log(-torch.log(noise))
    #     return torch.argmax(logits + gumbel_noise, dim=-1)

    # def q_posterior_logits(self, x_0, x_t, t):
    #     # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
    #     # otherwise, we return the L_{t-1} loss.
    #     # Also, we never have t == 0.

    #     # if x_0 is integer, we convert it to one-hot.
    #     if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
    #         x_0_logits = torch.log(
    #             torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
    #         )
    #     else:
    #         x_0_logits = x_0.clone()

    #     assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
    #         f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
    #     )

    #     # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

    #     # fact1 is "guess of x_{t-1}" from x_t
    #     # fact2 is "guess of x_{t-1}" from x_0

    #     fact1 = self._at(self.q_one_step_transposed, t, x_t)

    #     softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
    #     qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
    #     # bs, num_classes, num_classes
    #     fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

    #     out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

    #     t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

    #     bc = torch.where(t_broadcast == 1, x_0_logits, out)

    #     return bc

    # def vb(self, dist1, dist2):

    #     # flatten dist1 and dist2
    #     dist1 = dist1.flatten(start_dim=0, end_dim=-2)
    #     dist2 = dist2.flatten(start_dim=0, end_dim=-2)

    #     out = torch.softmax(dist1 + self.eps, dim=-1) * (
    #         torch.log_softmax(dist1 + self.eps, dim=-1)
    #         - torch.log_softmax(dist2 + self.eps, dim=-1)
    #     )
    #     return out.sum(dim=-1).mean()

    # def q_sample(self, x_0, t, noise):
    #     # forward process, x_0 is the clean input.
    #     logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
    #     noise = torch.clip(noise, self.eps, 1.0)
    #     gumbel_noise = -torch.log(-torch.log(noise))
    #     return torch.argmax(logits + gumbel_noise, dim=-1)

    # def model_predict(self, x_0, t, cond):
    #     # this part exists because in general, manipulation of logits from model's logit
    #     # so they are in form of x_0's logit might be independent to model choice.
    #     # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
    #     # they introduce at appendix A.8.

    #     predicted_x0_logits = self.x0_model(x_0, t, cond)

    #     return predicted_x0_logits

    # def forward_legacy(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
    #     """
    #     Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
    #     x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
    #     """
    #     t = torch.randint(1, self.T, (x.shape[0],), device=x.device)
    #     x_t = self.q_sample(
    #         x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
    #     )
    #     # x_t is same shape as x
    #     assert x_t.shape == x.shape, print(
    #         f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
    #     )
    #     # we use hybrid loss.

    #     predicted_x0_logits = self.model_predict(x_t, t, cond)

    #     # based on this, we first do vb loss.
    #     true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
    #     pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

    #     vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

    #     predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
    #     x = x.flatten(start_dim=0, end_dim=-1)

    #     ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

    #     return self.hybrid_loss_coeff * vb_loss + ce_loss, {
    #         "vb_loss": vb_loss.detach().item(),
    #         "ce_loss": ce_loss.detach().item(),
    #     }

    # def p_sample(self, x, t, cond, noise):

    #     predicted_x0_logits = self.model_predict(x, t, cond)
    #     pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

    #     noise = torch.clip(noise, self.eps, 1.0)

    #     not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

    #     gumbel_noise = -torch.log(-torch.log(noise))
    #     sample = torch.argmax(
    #         pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
    #     )
    #     return sample

    # def sample(self, x, cond=None):
    #     for t in reversed(range(1, self.T)):
    #         t = torch.tensor([t] * x.shape[0], device=x.device)
    #         x = self.p_sample(
    #             x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
    #         )

    #     return x

    # def sample_with_image_sequence(self, x, cond=None, stride=10):
    #     steps = 0
    #     images = []
    #     for t in reversed(range(1, self.T)):
    #         t = torch.tensor([t] * x.shape[0], device=x.device)
    #         x = self.p_sample(
    #             x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
    #         )
    #         steps += 1
    #         if steps % stride == 0:
    #             images.append(x)

    #     # if last step is not divisible by stride, we add the last image.
    #     if steps % stride != 0:
    #         images.append(x)

    #     return images


if __name__ == "__main__":
    import hydra

    @hydra.main(
        version_base=None,
        config_path="/mnt/CV_teamz/users/h1t/code/DiD/configs",
        config_name="config",
    )
    def main(config):
        model = CategoricalDiffusion(config)
        pseudo_input = torch.randn(2, config.model.vae_dim, 256, 256)
        out = model(pseudo_input)
        return out

    out = main()

    print(out)
