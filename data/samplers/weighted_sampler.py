import torch
from torch.utils.data.sampler import Sampler

from . import register_sampler


@register_sampler('weighted_sampler')
class WeightedSampler(Sampler):
    def __init__(self, initial_weights, batch_size, batches_per_iteration=None):
        """
        Randomly samples across dataset indices with provided weights
        Parameters
        ----------
        initial_weights : Iterable
            iterable obj with weights for each idx in dataset
        batch_size : int
            size of the returned batch
        batches_per_iteration: optional(int)
            number of batches per iteration. If None, will sample whole dataset
        """

        super().__init__([])
        self.weights = torch.tensor(initial_weights, dtype=torch.float)
        self.batch_size = batch_size
        self.batches_per_iteration = batches_per_iteration
        if self.batches_per_iteration:
            self.num_iterations = self.batches_per_iteration
        else:
            self.num_iterations = len(self.weights) // self.batch_size
        self.replacement = True if self.batch_size * self.num_iterations > len(self.weights) else False
    def __len__(self):
        return self.batch_size * self.num_iterations

    def __iter__(self):
        samples_num = self.batch_size * self.num_iterations
        rand_tensor = torch.multinomial(
            self.weights,
            samples_num,
            replacement=self.replacement,
        )
        return iter(rand_tensor.tolist())
