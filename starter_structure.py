import pysga
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import function
from pysga import Geometry, Structure
import pandas as pd
import settings
import os
from datetime import datetime
import itertools

def start_structure(AlfaMin=0.075, AlfaInitial=1, NIterations=700, GlobalIterationsRatio=0.6, PopulationSize=85,
SearchGroupRatio=0.20, NPerturbed=4, TournamentSize=5, SmallValue=0.0025, penal_topo=1e5, penal_others=1e8, struc_number=6):
    #INSERINDO DADOS NA GEOMETRIA
    geo = pysga.Geometry(struc_number=struc_number)

    #CRIANDO FUNÇÃO STRUC COM BASE NA GEOMETRIA
    struc = pysga.Structure(geo, penal_topo = penal_topo, penal_others = penal_others)

    #DEFININDO OS PARÂMETROS DO SGA:
    sga_params = pysga.ParamsSGA(AlfaMin=AlfaMin, AlfaInitial=AlfaInitial, NIterations=NIterations, GlobalIterationsRatio=GlobalIterationsRatio,
                 PopulationSize=PopulationSize, SearchGroupRatio=SearchGroupRatio, NPerturbed=NPerturbed)

    x, y = pysga.run(sga=sga_params, fobj=struc)
    print(struc.cost)
    return x, y

def test_parameters():

    struc_number = 6

    AlfaMin_list = [0.075]
    AlfaInitial_list = [1]
    GlobalIterationsRatio_list = [0.6]
    NIterations_list = [700]
    #PopulationSize_list = [76]
    NPerturbed_list = [4]
    TournamentSize_list = [5]
    penal_topo_list = [1e6]
    penal_others_list = [18, 20]

    combined = list(itertools.product(AlfaMin_list, AlfaInitial_list, GlobalIterationsRatio_list, NIterations_list,
    NPerturbed_list, TournamentSize_list, penal_topo_list, penal_others_list))

    print(f'Combinações: {len(combined)}')

    df_param_results = pd.DataFrame(columns = ['AlfaMin', 'AlfaInitial', 'GlobalIterationsRatio', 'NIterations', 'NPerturbed', 'TournamentSize', 'penal_topo',
    'penal_others', 'PopulationSize', 'y_min', 'y_avg', 'datetime'])

    # restart = int(input('Começando de qual iteração? '))-1
    restart = 0
    test_y_min = 1e10

    try:
        for k in range(restart, len(combined)):

            this_try = combined[k]

            print('-'*40)
            print(f'ENTRANDO NO TESTE Nº{k+1} DE PARÂMETROS')
            print('-'*40)

            AlfaMin_try = this_try[0]
            AlfaInitial_try = this_try[1]
            GlobalIterationsRatio_try = this_try[2]
            NIterations_try = this_try[3]
            NPerturbed_try = this_try[4]
            TournamentSize_try = this_try[5]
            penal_topo_try = this_try[6]
            penal_others_try = this_try[7]
            #PopulationSize_try = this_try[8]

            # PopulationSize_try = round((8000/NIterations_try)*1.23076)
            # PopulationSize_try = round((50000/NIterations_try)*1.19)
            PopulationSize_try = 85

            settings.init() #iniciando as variáveis globais
            y_sum = 0
            y_min = 10**10
            x_min = []
            NIR = 100

            df_y_results = pd.DataFrame()
            df_x_results = pd.DataFrame()
            df_xy_generations = pd.DataFrame()
            try:
                for i in range(NIR):
                    x, y = start_structure(AlfaMin=AlfaMin_try, AlfaInitial=AlfaInitial_try,  GlobalIterationsRatio=GlobalIterationsRatio_try,
                    NIterations=NIterations_try, PopulationSize=PopulationSize_try, NPerturbed=NPerturbed_try, TournamentSize=TournamentSize_try,
                    penal_topo=penal_topo_try, penal_others=penal_others_try, struc_number = struc_number)

                    print('-'*20+f'Rodagem Nº {i+1}, Teste Nº {k+1}'+'-'*20)
                    print(x.tolist())
                    print(y)
                    print('-'*60)
                    y_sum += y

                    if y < y_min:
                        y_min = y #tirando y_min
                        x_min = x.copy() #tirando x_min

                    if y < test_y_min:
                        test_y_min = y

                    df_y_results[str(i+1)] = settings.y_results
                    df_x_results[str(i+1)] = settings.x_results
                    df_xy_generations[str(i+1)] = settings.xy_generations

                    settings.y_results = list() #zerando para nova iteração
                    settings.x_results = list() #zerando para nova iteração
                    settings.xy_generations = list() #zerando para nova iteração
                    

            except:
                path = os.path.abspath('.')
                input_date = datetime.now().strftime("%d-%m-%Y-%H-%M")
                df_y_results.to_csv(path+'/results/problem'+str(struc_number)+'/y_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')
                df_x_results.to_csv(path+'/results/problem'+str(struc_number)+'/x_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')
                df_xy_generations.to_csv(path+'/results/problem'+str(struc_number)+'/xy_generations-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')
                fdsad

            path = os.path.abspath('.')
            input_date = datetime.now().strftime("%d-%m-%Y-%H-%M")
            df_y_results.to_csv(path+'/results/problem'+str(struc_number)+'/y_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')
            df_x_results.to_csv(path+'/results/problem'+str(struc_number)+'/x_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')
            df_xy_generations.to_csv(path+'/results/problem'+str(struc_number)+'/xy_generations-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+str(NIR)+"x--"+input_date+".csv", sep=',')

            y_avg = y_sum/NIR #Tirando a média

            set_params = {'AlfaMin': [AlfaMin_try], 'AlfaInitial': [AlfaInitial_try],
            'GlobalIterationsRatio': [GlobalIterationsRatio_try], 'NIterations': [NIterations_try],
            'NPerturbed': [NPerturbed_try], 'TournamentSize': [TournamentSize_try], 'penal_topo': [penal_topo_try],
            'penal_others': [penal_others_try], 'PopulationSize': [PopulationSize_try], 'y_min': [y_min], 'y_avg': [y_avg], 'datetime': [input_date]}

            df_set_params = pd.DataFrame(set_params)

            df_param_results = pd.concat([df_param_results, df_set_params], axis=0) #adicionando linha de resultados de parâmetros
            print(df_param_results)
            print(set_params)

        path = os.path.abspath('.')
        input_date = datetime.now().strftime("%d-%m-%Y-%H-%M")
        df_param_results.to_csv(path+'/results/problem'+str(struc_number)+'/param_results-problem'+str(struc_number)+"--"+str(round(test_y_min,2))+"kg--"+input_date+".csv", sep=',')

    except:
        print('')
        print('-------------------ERRO ENCONTRADO - CÓDIGO INTERROMPIDO COM BACKUP DE EMERGÊNCIA-------------------')
        print('')
        path = os.path.abspath('./param_results')
        input_date = datetime.now().strftime("%d-%m-%Y-%H-%M")
        df_param_results.to_csv(path+'/param_results-problem'+str(struc_number)+"--"+input_date+".csv", sep=',')

def optimize():
    settings.init()
    y_sum = 0
    y_min = 10**10
    x_min = []
    NIR = 150
    struc_number = 6

    df_y_results = pd.DataFrame()
    df_x_results = pd.DataFrame()

    for i in range(NIR):
        print('-'*30)
        print(f'ITERAÇÃO NIR = {i+1}')
        print('-'*30)

        x, y = start_structure(struc_number=struc_number)
        print(f'x = {x.tolist()}')
        print(f'y = {y}')
        y_sum += y
        df_y_results[str(i+1)] = settings.y_results
        df_x_results[str(i+1)] = settings.x_results
        if y < y_min:
            y_min = y
            x_min = x.copy()
        settings.y_results = list()
        settings.x_results = list()
    print(f'y_avg = {y_sum/NIR}')
    print(f'y_min = {y_min}')
    print(x_min.tolist())

    path = os.path.abspath('./csv_results')
    input_date = datetime.now().strftime("%d-%m-%Y-%H-%M")
    df_y_results.to_csv(path+'/y_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+input_date+".csv", sep=',')
    df_x_results.to_csv(path+'/x_results-problem'+str(struc_number)+"--"+str(round(y_min,2))+"kg--"+input_date+".csv", sep=',')

    pysga.plot_truss(np.array([x_min.tolist()]), struc_number)


if __name__ == '__main__':
    # optimize()
    for k in range(4):
        test_parameters()
