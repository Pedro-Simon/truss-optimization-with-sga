import os
import sys
path = os.path.abspath('..')
sys.path.append(path)

import pandas as pd
import numpy as np
from pysga import Geometry
from math import sqrt



def get_bests(problema):
    #Encontrando local da pasta
    mypath = os.path.abspath("../result_treatment/problem"+str(problema)+"/treated_results")

    #Coletando arquivos na pasta
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    this_files = ["", ""]

    for i in range(len(onlyfiles)):
        if onlyfiles[i].split("_")[0] == 'x':
            this_files[0] = onlyfiles[i]
        elif onlyfiles[i].split("_")[0] == 'y':
            this_files[1] = onlyfiles[i]

    x_df = pd.read_csv(mypath+'/'+this_files[0], index_col=0)
    y_df = pd.read_csv(mypath+'/'+this_files[1], index_col=0)
    
    x_best = list()

    for i in range(len(y_df)):
        y_min = 1e20
        for k in range(len(y_df.loc[0])):
            if float(y_df.iloc[i, k]) < y_min:
                y_min = float(y_df.iloc[i, k])
                j = k+0
        x_best.append(x_df.iloc[i, j][1:-1].split(', '))
    return np.array(x_best).astype(float)

def get_xy(problema):
    #Encontrando local da pasta
    mypath = os.path.abspath("../result_treatment/problem"+str(problema))

    #Coletando arquivos na pasta
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    total_file = [i for i in onlyfiles if i.split("_")[0] == 'xy']
    total_file.sort()

    for k in total_file:
        tabela = int(k.split("--")[-2][:-1])
        
    return mypath, total_file, tabela

def open_xy(file, mypath, necessary, done):
    csv = pd.read_csv(mypath+'/'+file, index_col=0).iloc[1:, 0:necessary]
    csv.reset_index(inplace=True, drop=True)
    csv.columns = [str(i) for i in range(done+1, done+len(csv.columns)+1)]
    return csv

def save_dindex(div_index, problema, weight, date, mypath, necessary):
    #Export para excel
    div_index.to_csv(mypath+'/'+'dindex_results'+'-problem'+str(problema)+"--"+str(weight)+"kg--"+str(necessary)+"x--"+str(date)+".csv", sep=',')

def calculate_dindex(df_xy, problema):
    #coletando lim_sup e lim_inf
    geo = Geometry(problema)
    lim_sup = geo.lim_sup
    lim_inf = geo.lim_inf

    #fazer o cálculo do diversity index para cada iteração
    div_index = dict()
    for k in range(len(df_xy.loc[0])):
        temp_diversity = np.zeros(len(df_xy))
        print(f'Coluna {k+1}')
        for i in range(len(df_xy)):
            temp = df_xy.iloc[i, k].split("[[")[1].split("]]")[0].split('], [')
            for f in range(len(temp)):
                temp[f] = temp[f].split(', ')
            x_iteration = np.array(temp).astype(float)[:, 1:]
            y_iteration = np.array(temp).astype(float)[:,:1]

            y_min = 1e10
            for y in range(len(y_iteration)):
                if y_iteration[y] < y_min:
                    y_min = y_iteration[y]+0
                    x_best = x_iteration[y]

            for n in range(x_iteration.shape[0]):
                temp_var = 0
                for p in range(x_iteration.shape[1]):
                    temp_var += ((x_best[p]-x_iteration[n, p])/(lim_sup[p] - lim_inf[p]))**2
                temp_diversity[i] += sqrt(temp_var)/(x_iteration.shape[0])
        div_index[str(k+1)] = temp_diversity
    div_index = pd.DataFrame(div_index)
    return div_index

def concat_dindex(problema):
    #Encontrando local da pasta
    mypath = os.path.abspath("../result_treatment/problem"+str(problema))

    #Coletando arquivos na pasta
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    #Alocando as pastas em cada local

    total_file = [i for i in onlyfiles if i.split("_")[0] == 'dindex']
    total_file.sort()

    date = total_file[0].split("--")[-1][:-4]
    weight = total_file[0].split("--")[1][:-2]

    csv1 = pd.read_csv(mypath+'/'+total_file[0], index_col=0)

    if len(total_file) > 1:
        csv2 = pd.read_csv(mypath+'/'+total_file[1], index_col=0)
        #Merge de dois csv's
        csv_final = pd.concat([csv1, csv2], axis=1)
    
    else:
        csv_final = csv1

    #Export para excel
    csv_final.to_csv(mypath+'/treated_results/'+'dindex_results'+'-problem'+str(problema)+"--"+str(weight)+"kg--100x--"+str(date)+".csv", sep=',')

def complete_treatment(problema):
    mypath, total_file, tabela = get_xy(problema) #Vai retornar quantos xy tiverem
    meta = 100
    done = 0
    
    for file in total_file:
        if tabela < meta:
            necessary = tabela + 0
        else:
            necessary = meta + 0

        #abrindo xy e atualizando o restante
        df_xy = open_xy(file, mypath, necessary, done)
        done += necessary
        meta -= necessary + 0

        #calculando o índice de diversidade
        div_index = calculate_dindex(df_xy, problema)

        df_xy = 0
        #salvando index na pasta
        date = file.split("--")[-1][:-4]
        weight = file.split("--")[1][:-2]
        save_dindex(div_index, problema, weight, date, mypath, necessary)

    concat_dindex(problema)


complete_treatment(5)