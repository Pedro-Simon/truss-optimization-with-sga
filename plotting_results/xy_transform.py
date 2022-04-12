import pandas as pd
import os

problema = 5

#Encontrando local da pasta
mypath = os.path.abspath("../result_treatment/problem"+str(problema))

#Coletando arquivos na pasta
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
print(onlyfiles)

#Alocando as pastas em cada local
total_files = dict()
weights = list()
dates = list()

for i in range(len(onlyfiles)):
    if onlyfiles[i].split("_")[0] == 'xy':
        date = onlyfiles[i].split("--")[-1][:-4]
        total_files[date] = ["", "", ""]
        dates.append(date)
        weights.append(float(onlyfiles[i].split("--")[1][:-2]))

for i in range(len(onlyfiles)):
    if onlyfiles[i].split("_")[0] == 'xy':
        date = onlyfiles[i].split("--")[-1][:-4]
        total_files[date][0] = onlyfiles[i]
    elif onlyfiles[i].split("_")[0] == 'x':
        date = onlyfiles[i].split("--")[-1][:-4]
        total_files[date][1] = onlyfiles[i]
    elif onlyfiles[i].split("_")[0] == 'y':
        date = onlyfiles[i].split("--")[-1][:-4]
        total_files[date][2] = onlyfiles[i]
    

print(total_files)
print(weights)


#Abrir o csv
if weights[0] <= weights[1]:
    i = 0
    j = 1
else:
    i = 1
    j = 0


#Lendo os csv's
list_type = ['xy_generations', 'x_results', 'y_results']
n_pertable = 70

for m in range(1, 3):
    print('Chegou')
    csv1 = pd.read_csv(mypath+'/'+total_files[dates[i]][m], index_col=0)

    csv2 = pd.read_csv(mypath+'/'+total_files[dates[j]][m], index_col=0).iloc[:, 0:(100-n_pertable)]
    csv2.columns = [str(i) for i in range(n_pertable+1, 101)]
    
    
    problema = int(total_files[dates[i]][m].split("--")[0][-1:])
    

    #Merge de dois csv's
    csv_final = pd.concat([csv1, csv2], axis=1)
    print(csv_final)

    #Export para excel
    csv_final.to_csv(mypath+'/treated_results/'+list_type[m]+'-problem'+str(problema)+"--"+str(weights[i])+"kg--100x--"+str(dates[i])+".csv", sep=',')