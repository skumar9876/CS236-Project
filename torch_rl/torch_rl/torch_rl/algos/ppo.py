import numpy
import torch
import torch.nn.functional as F

from torch_rl.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=1e-5, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, beta=1, use_l2w=False, sni_type=None, policy_loss_coef=1.0, 
                 reconstruction_likelihood_coef=0.0, KLD_coef=0.0, latent_transition_coef=0.0,
                 flow: bool = False):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.sni_type = sni_type
        self.policy_loss_coef = policy_loss_coef
        self.reconstruction_likelihood_coef = reconstruction_likelihood_coef
        self.KLD_coef = KLD_coef
        self.latent_transition_coef = latent_transition_coef
        self.flow = flow
        assert self.batch_size % self.recurrence == 0

        if use_l2w:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps, weight_decay=beta)
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_kl = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_kl = 0
                batch_loss = 0

                # Initialize memory


                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    # Use the current observation, action, and next observation to construct the latent transition loss.
                    transition_loss = 0
                    try:
                        obs = sb.obs
                        sb_next = exps[inds + i + 1]
                        next_obs = sb_next.obs
                        action = sb.action
                        if self.flow:
                            _, neg_log_prob = self.acmodel.transition_flow_inverse(obs, action, next_obs)
                            transition_loss = neg_log_prob.mean()
                        else:
                            next_latent_mean_pred, next_latent_log_var_pred = self.acmodel.transition_forward(obs, action)
                            next_latent_mean, _ = self.acmodel.vae_encode(next_obs)
                        
                            transition_loss = (next_latent_mean_pred - next_latent_mean)**2 / torch.exp(next_latent_log_var_pred)
                            transition_loss = transition_loss.mean()
                    except:
                        # Note: This except block is here for when the indices are already at the final observations in the batch, 
                        #       so there are no next observations to index.
                        pass

                    # Compute VAE loss
                    recon_obs_mu, recon_obs_logvar, mu, logvar = self.acmodel.vae_forward(sb.obs)
                    reconstruction_normal_dist = torch.distributions.normal.Normal(recon_obs_mu[:, :, :-1, :-1], torch.exp(0.5*recon_obs_logvar[:, :, :-1, :-1]))
                    image = torch.transpose(torch.transpose(sb.obs.image, 1, 3), 2, 3)
                    reconstruction_likelihood = reconstruction_normal_dist.log_prob(image).mean()

                    # see Appendix B from VAE paper:
                    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                    # https://arxiv.org/abs/1312.6114
                    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1).mean()

                    # Use the standard random policy and value function

                    dist, value = self.acmodel.compute_train(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()


                    loss = self.policy_loss_coef * policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss - self.reconstruction_likelihood_coef * reconstruction_likelihood + self.KLD_coef * KLD + self.latent_transition_coef * transition_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                grad_norm = 0.
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
