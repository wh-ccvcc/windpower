"""One-dimensional version of the MHCB local feature block."""

import torch
from torch import nn


class TemporalMHCB(nn.Module):
    """Parallel 3- and 5-point convolutions with two stages of feature mixing."""

    def __init__(self, channels):
        super().__init__()
        self.first_3 = nn.Conv1d(channels, channels, 3, padding=1)
        self.first_5 = nn.Conv1d(channels, channels, 5, padding=2)
        self.mix_3 = nn.Conv1d(3 * channels, channels, 1)
        self.mix_5 = nn.Conv1d(3 * channels, channels, 1)
        self.second_3 = nn.Conv1d(channels, channels, 3, padding=1)
        self.second_5 = nn.Conv1d(channels, channels, 5, padding=2)
        self.output = nn.Conv1d(5 * channels, channels, 1)

    def forward(self, x):
        local_3 = x + self.first_3(x).relu()
        local_5 = x + self.first_5(x).relu()
        mixed = torch.cat((x, local_3, local_5), dim=1)
        refined_3 = self.second_3(self.mix_3(mixed)).relu()
        refined_5 = self.second_5(self.mix_5(mixed)).relu()
        return self.output(torch.cat((mixed, refined_3, refined_5), dim=1))


class MHCBAdapter(nn.Module):
    def __init__(self, embedding_width, channels):
        super().__init__()
        self.input = nn.Conv1d(embedding_width, channels, 1)
        self.local = TemporalMHCB(channels)
        self.output = nn.Conv1d(channels, embedding_width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, embeddings):
        local = self.local(self.input(embeddings.transpose(1, 2)))
        return embeddings + self.output(local).transpose(1, 2)
