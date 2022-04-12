import os
import sys
path = os.path.abspath('..')
sys.path.append(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysga import Geometry, Structure, plot_truss

#Recolher todos os arquivos de treated_results
def get_files(problema):
#Encontrando local da pasta
    mypath = os.path.abspath("../result_treatment/problem"+str(problema)+"/treated_results")

    #Coletando arquivos na pasta
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    for i in onlyfiles:
        if i.split("_")[0] == 'x':
            x_name = i
        elif i.split("_")[0] == 'y':
            y_name = i
        elif i.split("_")[0] == 'dindex':
            dindex_name = i
    
    return x_name, y_name, dindex_name, mypath

#Criar um def para abrir pastas
def open_file(mypath, file_name):
    csv = pd.read_csv(mypath+'/'+file_name, index_col=0)
    return csv

#Criar def que pega os mínimos e média de y e dindex e o mínimo de x
def mean_minimum(x_df, y_df, dindex_df):

    #Encontrar o i em que está o mínimo de y
    i_sorted = np.argsort(np.array(y_df.iloc[-1, :].tolist()))
    
    x_sorted = list()
    for i in i_sorted:
        x_sorted.append(x_df.iloc[-1, :].tolist()[i])

    # #Pegar média e mínimo de y
    y_minimum = y_df.iloc[:, i_sorted[0]].tolist()
    y_mean = y_df.mean(axis=1).tolist()

    #Pegar média e mínimo de dindex
    dindex_minimum = dindex_df.iloc[:, i_sorted[0]].tolist()
    dindex_mean = dindex_df.mean(axis=1).tolist()

    return y_minimum, y_mean, dindex_minimum, dindex_mean, x_sorted

#Fazer o plot do gráfico de diversity (GENÉRICO)
def plot_graph(mypath, mean, minimum, problema, weight, date, type):

    if type == 'dindex':
        color = 'tab:orange'
    elif type == 'y':
        color = 'tab:blue'

    #Plot do gráfico
    plt.figure(figsize=(9, 5))
    plt.plot(mean, "grey", label="Média global", linestyle='dashed')
    plt.plot(minimum, color, label="Melhor Solução")
    plt.legend()
    plt.grid()

    #Arrumar legendas
    if type == 'dindex':
        plt.ylabel('Índice de Diversidade')
        #Título
        plt.title('Índice de Diversidade')
        #Definindo Nome
        name = 'dindex'
    elif type == 'y':
        plt.ylabel('Massa (kg)')
        #Título
        plt.title('Curva de convergência')
        #Definindo Nome
        name = 'y'

    plt.xlabel('Iterações')

    #Salvar para o caminho e nome correto
    plt.savefig(mypath+'/'+name+'-problem'+str(problema)+"--"+str(weight)+"kg--"+str(date)+'.png', bbox_inches='tight')

#Função que pega valores de mean_minimum e aplica para plotagem do gráfico e da estrutura
def plot_results(problema):
    x_name, y_name, dindex_name, mypath = get_files(problema)
    x_df = open_file(mypath, x_name)
    y_df = open_file(mypath, y_name)
    dindex_df = open_file(mypath, dindex_name)

    #Pegar o problema, weight e date
    date = x_name.split("--")[-1][:-4]
    weight = x_name.split("--")[1][:-2]

    y_minimum, y_mean, dindex_minimum, dindex_mean, x_sorted = mean_minimum(x_df, y_df, dindex_df)

    #Plot Diversity Index
    type = 'dindex'
    plot_graph(mypath, dindex_mean, dindex_minimum, problema, weight, date, type)

    #Plot Curva de Convergência
    type = 'y'
    plot_graph(mypath, y_mean, y_minimum, problema, weight, date, type)

    #Plot da estrutura
    for i in range(15):
        x = np.array([x_sorted[i][1:-1].split(', ')]).astype(float)
        path_name = mypath+'/'+str(i+1)+'st-structure'+'-problem'+str(problema)+"--"+str(weight)+"kg--"+str(date)+'.png'
        plot_truss(x, struc_number=problema, path_name = path_name, has_title=False, has_axis=False, is_horizontal=False, inside_text= False, y_space = -0.1)

plot_results(5)