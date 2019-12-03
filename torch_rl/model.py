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
                 dropout=0, use_l2a=False, use_bn=False, sni_type=None):
        super().__init__()

        # Decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.use_l2a = use_l2a
        self.dropout = dropout
        self.model_type = model_type
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
                nn.Conv2d(32, 64, 3, padding=1, stride=2)
            )

            self.image_deconv = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2, output_padding=1), 
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 6, 3, padding=1, stride=1)
            )

        # Not supported with current bottleneck

        # Resize image embedding
        self.embedding_size = 512

        # Define actor's model
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Unknown action space: " + str(action_space))
        
        self.reg_layer = nn.Linear(self.embedding_size, 64)

        self.actor = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(512, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        self.latent_transition = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=2))

        # Initialize parameters correctly
        self.apply(initialize_parameters)

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
        mean = embedding[:, :32]
        import pdb; pdb.set_trace()
        log_variance = embedding[:, 32:]

        return mean, log_variance

    def vae_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def vae_decode(self, z):
        reconstructed_img = self.image_deconv(z)
        reconstructed_img_mean = reconstructed_img[:, 0:3]
        reconstructed_img_variance = reconstructed_img[:, 3:]
        return reconstructed_img_mean, reconstructed_img_variance

    def vae_forward(self, obs):
        mu, logvar = self.vae_encode(obs)
        z = self.vae_reparameterize(mu, logvar)
        reconstructed_img_mean, reconstructed_img_variance = self.vae_decode(z)
        return reconstructed_img_mean, reconstructed_img_variance, mu, logvar

    # Latent transition model functions.
    def transition_forward(self, latent_obs, next_obs, action):
        next_image = torch.transpose(torch.transpose(next_obs.image, 1, 3), 2, 3)
        next_image_embedding = self.image_conv(next_image)
        mean = next_image_embedding[:, :32]
        log_variance = next_image_embedding[:, 32:]
        next_mean = self.latent_transition(mean)


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
