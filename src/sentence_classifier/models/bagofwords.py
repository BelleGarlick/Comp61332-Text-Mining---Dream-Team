from torch import nn
from torch import sum

import torch


class BagOfWords(nn.Module):

    def __init__(self):
        super(BagOfWords, self).__init__()

    def forward(self, x: torch.Tensor):
        num_words, batch_size, embedding_length = x.size()
        x = sum(x, 0) / num_words  # sums the cols of tensor x

        return x
