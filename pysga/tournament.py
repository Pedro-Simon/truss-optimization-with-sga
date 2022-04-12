# coding: utf-8

import numpy as np


class Tournament:

    def __init__(self, total, n_pertubed, size):
        self.total = int(total)
        self.n_pertubed = int(n_pertubed)
        self.size = int(size)
        self.range_tournament = range(self.n_pertubed)
        self.tournament_size_array = np.ones((self.size, 1), dtype=np.int64) * total
        self.perturbed_indices = np.zeros(n_pertubed, dtype=np.int64)
        self.indices = np.arange(1, total)

    def tournament(self):
        indices = self.indices.copy()
        for i in self.range_tournament:
            c = self.tournament_size_array - np.random.randint(1, self.total + 1 - i, size=(self.size, 1))
            k = c.min()
            self.perturbed_indices[i] = indices[k]
            try:
                indices[i:k + 1] = indices[i - 1:k]
            except ValueError:
                pass
        return self.perturbed_indices

    def reverse_tournament(self):
        indices = self.indices.copy()
        for i in self.range_tournament:
            c = self.tournament_size_array - np.random.randint(2, self.total + 1 - i, size=(self.size, 1))
            k = c.max()
            self.perturbed_indices[i] = indices[k]
            try:
                indices[i:k + 1] = indices[i - 1:k]
            except ValueError:
                try:
                    indices[i:] = indices[i:k]
                except:
                    pass
        return self.perturbed_indices
