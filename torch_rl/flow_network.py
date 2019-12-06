import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""
    def __init__(self, input_size, output_size, mask, extra_dim: int = None):
        super().__init__(input_size, output_size)
        self.register_buffer('mask', mask)
        self.extra_dim = extra_dim
        if extra_dim is not None:
            self.extra_weight = nn.Linear(extra_dim, output_size, bias=False)

    def forward(self, x, y=None):
        masked_out = F.linear(x, self.mask * self.weight, self.bias)
        if self.extra_dim is not None:
            y_out = self.extra_weight(y)
            return masked_out + y_out
        else:
            return masked_out

class PermuteLayer(nn.Module):
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """
    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device)

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    https://arxiv.org/abs/1502.03509

    Uses sequential ordering as in the MAF paper.
    Gaussian MADE to work with real-valued inputs"""
    def __init__(self, input_size, hidden_size, n_hidden, condition_dim: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0], extra_dim=condition_dim)]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(
                self.hidden_size, self.hidden_size, masks[i+1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        self.net += [MaskedLinear(
            self.hidden_size, self.input_size * 2, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees] # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden+1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]
        self.m = degrees

        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, z):
        """
        Run the forward mapping (z -> x) for MAF through one MADE block.
        :param z: Input noise of size (batch_size, self.input_size)
        :return: (x, log_det). log_det should be 1-D (batch_dim,)
        """
        x = torch.zeros_like(z)
    
        # YOUR CODE STARTS HERE
        for idx in range(self.input_size):
            theta = self.net(x)
            mu, std = theta[:,idx], theta[:,self.input_size + idx].exp()
            x[:,idx] = z[:,idx] * std + mu
        log_det = None
        # YOUR CODE ENDS HERE

        return x, log_det

    def inverse(self, x, y):
        """
        Run one inverse mapping (x -> z) for MAF through one MADE block.
        :param x: Input data of size (batch_size, self.input_size)
        :return: (z, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE
        theta = self.net[0](x, y)
        theta = self.net[1:](theta)
        mu, alpha = theta[:,:self.input_size], theta[:,self.input_size:]
        z = (x - mu) / alpha.exp()
        log_det = -alpha.sum(-1)
        # YOUR CODE ENDS HERE

        return z, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow, using MADE layers.
    https://arxiv.org/abs/1705.07057
    """
    def __init__(self, input_size, hidden_size, n_hidden, n_flows, condition_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.base_dist = torch.distributions.normal.Normal(0,1)
        
        # need to flip ordering of inputs for every layer
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(
                MADE(self.input_size, self.hidden_size, self.n_hidden, condition_dim=condition_dim))
            nf_blocks.append(PermuteLayer(self.input_size))     # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def log_probs(self, x, y):
        """
        Obtain log-likelihood p(x) through one pass of MADE
        :param x: Input data of size (batch_size, self.input_size)
        :return: log_prob. This should be a Python scalar.
        """
        # YOUR CODE STARTS HERE
        z = x
        log_det_sum = 0
        for idx, flow in enumerate(self.nf):
            if isinstance(flow, PermuteLayer):
                z, log_det = flow.inverse(z)
            else:
                z, log_det = flow.inverse(z, y)
            log_det_sum += log_det.squeeze()
        log_prob = (self.base_dist.log_prob(z).sum(-1) + log_det_sum).mean()
        # YOUR CODE ENDS HERE

        return log_prob

    def loss(self, x):
        """
        Compute the loss.
        :param x: Input data of size (batch_size, self.input_size)
        :return: loss. This should be a Python scalar.
        """
        return -self.log_probs(x)

    def sample(self, device, n):
        """
        Draw <n> number of samples from the model.
        :param device: [cpu,cuda]
        :param n: Number of samples to be drawn.
        :return: x_sample. This should be a numpy array of size (n, self.input_size)
        """
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample)
            x_sample = x_sample.view(n, self.input_size)
            x_sample = x_sample.cpu().data.numpy()

        return x_sample
