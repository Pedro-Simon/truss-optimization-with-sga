# coding: utf-8

import numpy as np


class ObjectiveFunc:

    def __init__(self, inf=[-10., -10.], sup=[10., 10.]):
        self.inf = np.array(inf)
        self.sup = np.array(sup)
        self.dim = self.inf.shape[0]
        self.range_5 = range(5)
        self.cos = np.cos
        self.zeros = np.zeros

    def evaluate(self, x):
        y = self.zeros((x.shape[0], 1))
        s1 = self.zeros(x.shape[0])
        s2 = self.zeros(x.shape[0])
        for i in self.range_5:
            n = i + 1
            s1 = s1 + n * self.cos((n + 1) * x[:, 0] + n)
            s2 = s2 + n * self.cos((n + 1) * x[:, 1] + n)
        y[:, 0] = s1 * s2
        return y


class ParamsSGA:
    def __init__(self, AlfaMin=0.01, AlfaInitial=2, NIterations=500, GlobalIterationsRatio=0.3,
                 PopulationSize=100, SearchGroupRatio=0.1, NPerturbed=5, TournamentSize=5,
                 SmallValue=0.0025):
        self.AlfaMin = AlfaMin
        self.AlfaInitial = AlfaInitial
        self.NIterations = NIterations
        self.GlobalIterationsRatio = GlobalIterationsRatio
        self.PopulationSize = PopulationSize
        self.SearchGroupRatio = SearchGroupRatio
        self.NPerturbed = int(NPerturbed)
        self.PlotFamily = False
        self.TournamentSize = int(TournamentSize)
        self.elite = 1
        self.SmallValue = SmallValue
        self.alfa_local = False
        self.current_iteration = 0
        self.NIterationsGlobal = int(self.GlobalIterationsRatio*self.NIterations)
        self.NIterationsLocal = self.NIterations - self.NIterationsGlobal
        self.AlfaCorrectionMatrix = np.array([[1, -4 / (self.NIterationsGlobal)],
                                              [0.25, -1 / (4 * self.NIterationsGlobal)],
                                              [0, 0]])

    def update_iterarion(self):
        self.current_iteration += 1

    def update_AlfaCorrectionMatrix(self):
        self.AlfaCorrectionMatrix = np.array([[1, -4 / (self.NIterationsGlobal)],
                                         [0.25, -1 / (4 * self.NIterationsGlobal)],
                                         [0, 0]])

    def alfa(self, inf, sup):
        if self.alfa_local:
            _alfa = (self.NIterationsLocal - self.current_iteration)/(self.NIterationsLocal)
            _alfa *= self.AlfaMin
            _alfa += (self.SmallValue * self.AlfaMin)
        else:
            _alfa = (self.AlfaInitial * self.alfa_correction() + self.AlfaMin)
        return _alfa * (sup - inf)

    def alfa_correction(self):
        if self.current_iteration == 0:
            return 0
        v = self.AlfaCorrectionMatrix[:, 0] + self.AlfaCorrectionMatrix[:, 1] * self.current_iteration
        return v.max()


def configure():
    f = ObjectiveFunc()
    sga = ParamsSGA()
    return sga, f


if __name__ == '__main__':
    sga, fobj = configure()
    z = fobj.evaluate(np.array([[100*0.048579216311839, 100*0.054828050124931]]))
    print(z)
