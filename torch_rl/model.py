import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym
from bottleneck import Bottleneck

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self, obs_space, action_space, model_type="default", use_bottleneck=False,
                 dropout=0, use_l2a=False, use_bn=False, sni_type=None, flow: bool = False,
                 flow_depth: int = 3, num_latent_channels: int = 32):
        super().__init__()

        # Decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.use_l2a = use_l2a
        self.dropout = dropout
        self.model_type = model_type
        self.flow = flow
        self.flow_depth = flow_depth
        self.action_space = action_space
        self.latent_dim = num_latent_channels
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        print(n,m)

        # Define image embedding
        if model_type == "default2":
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, self.latent_dim * 2, 3, padding=1, stride=2)
            )

            self.image_deconv = nn.Sequential(
                nn.ConvTranspose2d(self.latent_dim, 32, 3, padding=1, stride=2, output_padding=1), 
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 6, 3, padding=1, stride=1)
            )

            self.num_actions = action_space.n
            self.latent_transition = nn.Sequential(nn.Conv2d(self.latent_dim, self.latent_dim * 2 * self.num_actions, 3, padding=1, stride=1))

        # Not supported with current bottleneck

        # Resize image embedding    
        self.embedding_size = self.latent_dim * 4 * 4

        print(flow)
        if flow:
            self.flow_estimator = nn.Sequential(
                nn.Linear(self.embedding_size + action_space.n, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, flow_depth * self.embedding_size * (self.embedding_size + 1) // 2)
            )
        
        # Define actor's model
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Unknown action space: " + str(action_space))
        
        self.reg_layer = nn.Linear(self.embedding_size, 64)

        self.actor = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)
        print(self.latent_transition)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        raise NotImplementedError

    # VAE functions. 
    def vae_encode(self, obs):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        embedding = self.image_conv(x)
        mean = embedding[:, :self.latent_dim]
        log_variance = embedding[:, self.latent_dim:]
        return mean, log_variance

    def vae_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def vae_decode(self, z):
        reconstructed_img = self.image_deconv(z)
        reconstructed_img_mean = reconstructed_img[:, :3]
        reconstructed_img_variance = reconstructed_img[:, 3:]
        return reconstructed_img_mean, reconstructed_img_variance

    def vae_forward(self, obs):
        mu, logvar = self.vae_encode(obs)
        z = self.vae_reparameterize(mu, logvar)
        reconstructed_img_mean, reconstructed_img_variance = self.vae_decode(z)
        return reconstructed_img_mean, reconstructed_img_variance, mu, logvar

    # Latent transition model functions.
    def transition_forward(self, obs, action):
        latent_mean, latent_log_var = self.vae_encode(obs)
        latent_obs = latent_mean

        next_latent_embeddings_pred = self.latent_transition(latent_obs)
        next_latent_embeddings_pred = next_latent_embeddings_pred.view(-1, self.num_actions, 2 * self.latent_dim, 4, 4)
        next_latent_embeddings_pred = next_latent_embeddings_pred.gather(1, action.long().view(-1, 1, 1, 1, 1).expand(next_latent_embeddings_pred.shape[0], 1, 2 * self.latent_dim, 4, 4)).squeeze()
        next_latent_mean_pred = next_latent_embeddings_pred[:, :self.latent_dim]
        next_latent_log_var_pred = next_latent_embeddings_pred[:, self.latent_dim:]

        return next_latent_mean_pred, next_latent_log_var_pred

    def _get_flow_weights(self, latent_obs, action, inverse: bool = False):
        one_hot_action = torch.zeros(latent_obs.shape[0], self.action_space.n, device=latent_obs.device)
        one_hot_action[torch.arange(latent_obs.shape[0], device=latent_obs.device).long(), action.long()] = 1
        flow_theta = self.flow_estimator(torch.cat((latent_obs.view(latent_obs.shape[0], -1), one_hot_action), -1)).view(latent_obs.shape[0] * self.flow_depth, -1)
        idxs = torch.tril_indices(self.embedding_size, self.embedding_size, device=latent_obs.device).repeat(1, latent_obs.shape[0] * self.flow_depth)
        batch_idxs = torch.arange(latent_obs.shape[0], device=latent_obs.device).unsqueeze(0).repeat(flow_theta.shape[-1] * self.flow_depth, 1).permute(1,0).reshape(-1).long()
        W = torch.zeros(latent_obs.shape[0] * self.flow_depth, self.embedding_size, self.embedding_size, device=latent_obs.device)
        W[batch_idxs,idxs[0],idxs[1]] = flow_theta.view(-1)
        batch_diag = torch.eye(W.shape[-1], device=W.device).unsqueeze(0).repeat(W.shape[0], 1, 1).bool()
        W[batch_diag] = W[batch_diag].exp()
        if inverse:
            W = W.inverse()
        W = W.view(latent_obs.shape[0], self.flow_depth, self.embedding_size, self.embedding_size)
        return W

    def _inv_lr(self, y, coef: float = 0.1):
        y[y<0] = y[y<0] / coef
    
    def transition_flow_forward(self, obs, action, lr_coef: float = 0.1):
        latent_obs, _ = self.vae_encode(obs)
        W = self._get_flow_weights(latent_obs, action)
        z = torch.empty_like(latent_obs.view(latent_obs.shape[0], -1)).normal_()
        for idx in range(self.flow_depth):
            z = torch.bmm(W[:,idx], z.unsqueeze(-1)).squeeze(-1)
            if idx < self.flow_depth - 1:
                z = F.leaky_relu(z, lr_coef)
            
        return z.view(latent_obs.shape)

    def transition_flow_inverse(self, obs, action, next_obs, lr_coef: float = 0.1):
        latent_obs, _ = self.vae_encode(obs)
        next_latent_obs, _ = self.vae_encode(next_obs)
        W = self._get_flow_weights(latent_obs, action, inverse=True)
        x = next_latent_obs.view(next_latent_obs.shape[0], -1)
        log_det = torch.zeros(x.shape[0], device=x.device)
        for idx in reversed(range(self.flow_depth)):
            if idx < self.flow_depth - 1:
                self._inv_lr(x)

            act_jacob = torch.ones_like(x)
            act_jacob[x<0] = 1/lr_coef
            log_det += act_jacob.log().sum(-1)
            
            x = torch.bmm(W[:,idx], x.unsqueeze(-1)).squeeze(-1)

            weights = W[:,idx]
            eye = torch.eye(weights.shape[-1], device=W.device).unsqueeze(0).repeat(weights.shape[0], 1, 1).bool()
            diag = weights[eye].view(weights.shape[0], self.embedding_size)
            w_log_det = diag.abs().log().sum(-1)

            log_det += w_log_det

        log_prob = torch.distributions.Normal(torch.zeros(self.embedding_size, device=x.device),
                                              torch.ones(self.embedding_size, device=x.device)).log_prob(x)

        return x, -(log_prob.sum(-1) + log_det).mean()
    
    # RL agent functions.
    def encode(self, obs):
        embedding, _ = self.vae_encode(obs)
        embedding = embedding.reshape(embedding.shape[0], -1)
        return embedding

    def compute_run(self, obs):
        embedding = self.encode(obs)
        x_dist = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
        value = self.critic(embedding).squeeze(1)
        return dist, value

    def compute_train(self, obs):
        embedding = self.encode(obs)
        # Random policy AND value function
        x_dist_train = self.actor(embedding)
        dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))
        value = self.critic(embedding).squeeze(1)
        return dist_train, value
