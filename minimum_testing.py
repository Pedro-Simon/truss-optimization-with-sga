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

#Selecionar o problema
struc_number = 3

#Pegar o menor valor possível
# x = np.array([[6.155, 6.155, 6.155, 13.819, 8.6, 0.91, 1.742, 3.478, 2.839, 2.839, 0, 1.742, 0, 0, 11.381, 322.4111, 561.5408, 254, 254, 139.9969, -9.4016, 50.8, 138.7773]])

# x = np.array([[0, 74.1934, 18.58061, 37.03218, 74.1934, 46.58055, 0, 0, 87.0966, 0, 0, 20.057, 12.384, 13]])

x = np.array([[1.5249635551927319, 23.90956066611839, 10.852431889173141, 20.363967995578772, 22.913179426516624, 20.681060259072574, 1.6838580829611272, 0.12315587887480371, 23.61595285250534, 0.4595897446918739, 0.8766720179768291, 19.602942857200294, 12.276035811317074, 5.543982488505606]])
x = np.array([[15.709696607854287, 16.47290711650187, 14.27986584310238, 22.202484895414237, 16.77727285161069, 9.152944192436088, 10.98304852945968, 13.23658328355624, 16.595729217900644, 19.15006425383849, 12.195856067000143, 2.2086358000608355, 21.83448469870643, 4.180038660401839, 18.038610150368193, 2.9797299809272038, 6.214696507341382, 3.1442680408557035, 3.4135629745115583, 1.3807220301697483, 0.454097677426672, 0.051348141457099254, 1.4973645413552772]])

#Rodar o geo
geo = Geometry(struc_number)

#Converter reais em discretos (se houver)
x_size = x[0][:geo.n_el].copy()
x_shape = x[0][geo.n_el:].copy()
for i in range(len(geo.bars_list)):
    for j in range(len(x_size)):
        if geo.bars_list[i] == x_size[j]:
            x_size[j] = i+1

x = np.array([np.append(x_size, x_shape)])
print(x[0].tolist())

#Comparar para ver se passa nos boundaries
for i in range(len(x[0])):
    if geo.lim_inf[i] > x[0][i]:
        print(f'Limite inferior {geo.lim_inf[i]} extrapolado por {x[0][i]}')
    if geo.lim_sup[i] < x[0][i]:
        print(f'Limite superir {geo.lim_sup[i]} extrapolado por {x[0][i]}')


#Rodar o estrutural
struc = Structure(geo)
y = struc.evaluate_solo(x)
print(y)
print(struc.this_coord_nodes)
print(struc.this_conec)
struc.print_results()

#Comparar para ver se passa nas restrições (constraints)
#plotar estrutura
pysga.plot_truss(x, struc_number)
