import torch
import torch.nn as nn
import numpy as np


class PreNormException(Exception):
    pass


class PreNormLayer(nn.Module):
    def __init__(self, n_units, shift=True, scale=True):
        super().__init__()
        assert shift or scale

        if shift:
            self.shift = nn.Parameter(nn.init.constant_(torch.empty(n_units, ), 1), requires_grad=False).cuda()
        else:
            self.shift = None

        if scale:
            self.scale = nn.Parameter(nn.init.constant_(torch.empty(n_units, ), 1), requires_grad=False).cuda()
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input):
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input = input + self.shift

        if self.scale is not None:
            input = input * self.scale

        return input

    def update_stats(self, input):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        # assert self.n_units == 1 or input.size()[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.size()[-1]}."

        input = input.reshape([-1, self.n_units])
        sample_avg = torch.mean(input, dim=0)
        sample_var = torch.mean((input - sample_avg) ** 2, dim=0)
        sample_count = input.numel() / self.n_units
        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift.data = -self.avg

        if self.scale is not None:
            self.var = torch.where(torch.eq(self.var, 0), torch.ones_like(self.var), self.var)  # NaN check trick
            self.scale.data = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
