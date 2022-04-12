# coding: utf-8

import numpy as np
from pysga.family import Family
from pysga.configure import ParamsSGA, ObjectiveFunc
from pysga.tournament import Tournament
import matplotlib.pyplot as plt
import settings


def leaders_indices(f, sga, tournament):
    indices = np.transpose(np.arange(0, f.search_group.size, 1))
    tournament_indices = tournament.tournament()
    tournament_indices = np.sort(tournament_indices)
    indices[np.arange(sga.elite, f.search_group.size)] = tournament_indices
    return indices


def run(sga=None, fobj=None):
    if not sga:
        sga = ParamsSGA()
    if not fobj:
        fobj = ObjectiveFunc()
    f = Family(sga, fobj)
    n_pertubed = int(f.search_group.size - sga.elite)
    tournament = Tournament(f.population.size, n_pertubed, sga.TournamentSize)
    rtournament = Tournament(f.search_group.size, sga.NPerturbed, sga.TournamentSize)
    f.update_leaders(leaders_indices(f, sga, tournament))
    iteration_data = np.zeros((sga.NIterations, f.dim + 1))

    for k in range(sga.NIterationsGlobal):
        perturbed_indices = rtournament.reverse_tournament()
        f.mutate_leaders(perturbed_indices)
        f.generation(sga.alfa(fobj.inf, fobj.sup))
        f.sort_leaders()
        iteration_data[k] = f.leaders[0]
        # if (sga.current_iteration+1) % 50 == 0:
            # print("///////////////")
            # print(f'ITERAÇÃO GLOBAL: {sga.current_iteration+1}')
            # print(f'x = {f.leaders[0, 1:].tolist()}')
            # print(f'y = {f.leaders[0, 0]}')
        settings.x_results.append(str(f.leaders[0, 1:].tolist()))
        settings.y_results.append(f.leaders[0, 0])
        # try:
             # plt.plot(f.leaders[0,1], f.leaders[0,2], marker=11, markersize=3, c="y")
        # except:
        #     pass
        sga.update_iterarion()

    sga.alfa_local = True
    sga.current_iteration = 0

    for k in range(sga.NIterationsLocal):
        perturbed_indices = rtournament.reverse_tournament()
        f.mutate_leaders(perturbed_indices)
        f.local_generation(sga.alfa(fobj.inf, fobj.sup))
        f.sort_database()
        iteration_data[sga.NIterationsGlobal + k] = f.database[0]
        # try:
        #     plt.plot(f.database[0,1], f.database[0,2], marker="*", markersize=10, c="b")
        # except:
        #     pass
        f.update_leaders(leaders_indices(f, sga, tournament))
        # if (sga.NIterationsGlobal+sga.current_iteration+1) % 50 == 0:
            # print("///////////////")
            # print(f'ITERAÇÃO LOCAL: {sga.NIterationsGlobal+sga.current_iteration+1}')
            # print(f'x = {iteration_data[sga.NIterationsGlobal+k,1:].tolist()}')
            # print(f'FASE LOCAL = {iteration_data[sga.NIterationsGlobal+k,0]}')
        settings.x_results.append(str(iteration_data[sga.NIterationsGlobal+k,1:].tolist()))
        settings.y_results.append(iteration_data[sga.NIterationsGlobal+k,0])
        sga.update_iterarion()

    x = iteration_data[-1, 1:]
    y = iteration_data[-1, 0]
    return x, y


if __name__ == '__main__':
    x, y = run()
    print(x, y)
