# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import settings

class Family:

    def __init__(self, sga, fobj):
        self.inf = fobj.inf
        self.sup = fobj.sup
        self.dim = fobj.dim
        self.fertility = fobj.evaluate
        self.population = self.Population(sga)
        self.search_group = self.SearchGroup(sga)
        self.family_size = self._define_groups_amounts()
        self.unit_family_size = []
        self.family_data_indices = []
        self.initial_family_cte()
        self.leaders = np.zeros([self.search_group.size, self.dim+1])
        self.database = None
        self.initial_database()
        self.family = []
        self.initial_family()
        self.xTemp = []
        self.initial_xtemp()
        self.pertubed_arange = np.repeat(np.arange(1, sga.NPerturbed+1, 1),
                                         self.dim).reshape(sga.NPerturbed, self.dim)

    def initial_family_cte(self):
        for i in self.search_group.range:
            self.unit_family_size.append(np.ones((self.family_size[i], 1)))
            indices = np.arange(0, self.family_size[i]+1) + self.partial_population_families(i)
            self.family_data_indices.append(indices)

    def initial_xtemp(self):
        for i in self.search_group.range:
            self.xTemp.append(np.zeros([self.family_size[i], self.dim]))

    def initial_family(self):
        self.family = []
        for i in self.search_group.range:
            self.family.append(np.zeros((self.family_size[i]+1, self.dim+1)))

    def initial_database(self):
        inf = np.tile(self.inf, (self.population.size, 1))
        sup = np.tile(self.sup, (self.population.size, 1))
        x = inf + (sup - inf) * np.random.random((self.population.size, self.dim))
        y = self.fertility(x)
        self.database = np.hstack((y, x))
        settings.xy_generations = np.array([str(self.database.tolist())])
        self.database = self.database[np.argsort(self.database[:, 0])]

    def update_leaders(self, indices):
        self.leaders = self.database[indices]

    def sort_leaders(self):
        self.leaders = self.leaders[np.argsort(self.leaders[:, 0])]

    def sort_database(self):
        self.database = self.database[np.argsort(self.database[:, 0])]

    def mutate_leaders(self, pertubed_indices):
        mutation = np.mean(self.leaders[:, 1:], axis=0) * np.ones((len(pertubed_indices), self.dim))
        mutation += np.std(self.leaders[:, 1:], axis=0, ddof=1) * \
                    self.pertubed_arange * (np.random.random((len(pertubed_indices), self.dim)) - 0.5)
        np.maximum(mutation, self.inf, out=mutation)
        np.minimum(mutation, self.sup, out=mutation)
        self.leaders[pertubed_indices, 1:] = mutation

    def _define_groups_amounts(self):
            w = (np.arange(0, 1 + 1. / self.search_group.size, 1. / self.search_group.size) ** 2) \
                * (self.population.size - self.search_group.size)
            v = np.zeros(self.search_group.size)
            for i in range(self.search_group.size - 1):
                v[self.search_group.size - 1 - i] = max(round(w[i + 1] - w[i]), 1)
            v[0] = self.population.size - self.search_group.size - v.sum()
            return np.vectorize(int)(v)

    def partial_population_families(self, n):
        return int(self.family_size[:n].sum() + n)

    def generation(self, alfa):

        for i in self.search_group.range:
            self.xTemp[i] = self.leaders[i][1: self.dim+1] * self.unit_family_size[i] + \
                                                       alfa * self.unit_family_size[i] * \
                                                       (np.random.random((self.family_size[i], self.dim)) - 0.5)
            np.maximum(self.xTemp[i], self.inf, out=self.xTemp[i])
            np.minimum(self.xTemp[i], self.sup, out=self.xTemp[i])
            self.family[i][0, :] = self.leaders[i, :]
            self.family[i][1:self.family_size[i] + 1, 1: self.dim + 1] = self.xTemp[i]
            if i == 0:
                this_generation = self.xTemp[i]
            else:
                this_generation = np.append(this_generation, self.xTemp[i], axis=0)

        results_generation = self.fertility(this_generation)
        # for k in self.family[i][1:, 1: self.dim+1]:
        #     plt.plot(k[0], k[1], marker=11, markersize=8, c="y")
        before, after = 0, 0
        for i in self.search_group.range:
            after += self.family_size[i]
            self.family[i][1:, 0] = np.array([results_generation[:, 0][before:after]])
            before = after + 0
            values = self.family[i][:, 0]
            min_value_indice = values.argmin(axis=0)
            self.leaders[i] = self.family[i][min_value_indice, :]
            if i == 0:
                temp_xy_gen = self.family[i]
            else:
                temp_xy_gen = np.append(temp_xy_gen, self.family[i], axis=0)
        settings.xy_generations = np.append(settings.xy_generations, [str(temp_xy_gen.tolist())], axis=0)

    def local_generation(self, alfa):
        for i in self.search_group.range:
            self.xTemp[i] = self.leaders[i][1: self.dim+1] * self.unit_family_size[i] + \
                                                       alfa * self.unit_family_size[i] * \
                                                       (np.random.random((self.family_size[i], self.dim)) - 0.5)
            np.maximum(self.xTemp[i], self.inf, out=self.xTemp[i])
            np.minimum(self.xTemp[i], self.sup, out=self.xTemp[i])
            self.family[i][0, :] = self.leaders[i, :]
            self.family[i][1:self.family_size[i] + 1, 1: self.dim + 1] = self.xTemp[i]
            if i == 0:
                this_generation = self.xTemp[i]
            else:
                this_generation = np.append(this_generation, self.xTemp[i], axis=0)

            # for k in self.family[i][1:, 1: self.dim+1]:
            #     plt.plot(k[0], k[1], marker=11, markersize=8, c="b")
        results_generation = self.fertility(this_generation)
        # for k in self.family[i][1:, 1: self.dim+1]:
        #     plt.plot(k[0], k[1], marker=11, markersize=8, c="y")
        before, after = 0, 0
        for i in self.search_group.range:
            after += self.family_size[i]
            self.family[i][1:, 0] = np.array([results_generation[:, 0][before:after]])
            before = after + 0
            self.database[self.family_data_indices[i], :] = self.family[i][:]
            if i == 0:
                temp_xy_gen = self.family[i]
            else:
                temp_xy_gen = np.append(temp_xy_gen, self.family[i], axis=0)
        settings.xy_generations = np.append(settings.xy_generations, [str(temp_xy_gen.tolist())], axis=0)

    class Population:

        def __init__(self, sga):
            self.size = sga.PopulationSize

    class SearchGroup:

        def __init__(self, sga):
            self.ratio = sga.SearchGroupRatio
            self.size = int(round(self.ratio*sga.PopulationSize))
            self.range = range(self.size)
