import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


class Structure:
    #coord_nodes = matriz de coordenadas dos nós [n_node, coord_x, coord_y]
    #conec = matriz de conectividade []
    def __init__(self, geo, penal_topo = 1e20, penal_others = 1e20):
        self.cost = 0
        self.coord_nodes = geo.coord_nodes
        self.n_nodes = geo.n_nodes
        self.conec = geo.conec
        self.n_el = geo.n_el
        self.forces = geo.forces
        self.n_forces = geo.n_forces
        self.GDL_rest = geo.GDL_rest
        self.n_rest = geo.n_rest
        self.E = geo.mod_elast
        self.density = geo.density
        self.ten_adm = geo.ten_adm
        self.max_deflection = geo.max_deflection

        self.inf = geo.lim_inf
        self.sup = geo.lim_sup
        self.dim = self.inf.shape[0]
        self.range_5 = range(5)
        self.cos = np.cos
        self.zeros = np.zeros
        self.GDL = 2 * self.n_nodes

        self.is_topology = geo.is_topology
        self.is_discrete = geo.is_discrete
        if self.is_discrete == True:
            self.bars_list = geo.bars_list
        elif self.is_discrete == False and self.is_topology == True:
            self.min_topology = geo.min_topology
        self.is_shape = geo.is_shape
        if self.is_shape == True:
            self.n_shape_x = geo.n_shape_x
        self.shape_nodes_x = geo.shape_nodes_x
        self.shape_nodes_y = geo.shape_nodes_y
        self.is_buckling = geo.is_buckling

        if self.is_buckling == True:
            self.buckling = geo.buckling

        self.penal_topo = penal_topo
        self.penal_others = penal_others
        self.is_simetric = geo.is_simetric
        if self.is_simetric == True:
            self.size_vars = geo.size_vars

    def topology_correction(self):

        penalization = 0
        if self.is_simetric == True:
            self.x_size_temp = np.zeros(self.n_el)
            for i in range(len(self.size_vars)):
                for j in self.size_vars[i]:
                    self.x_size_temp[j-1] = self.x[i].copy()
            self.x = np.append(np.array(self.x_size_temp), self.x[len(self.size_vars):])

        if self.is_discrete == False and self.is_topology:
            for i in range(self.n_el):
                if self.x[i] <= self.min_topology:
                    self.x[i] = 0

        if self.is_discrete == True:
            self.this_x = np.append([self.bars_list[round(i)-1] if round(i) != 0 else 0 for i in self.x[:self.n_el]], self.x[self.n_el:])
        else:
            self.this_x = self.x.copy()
        self.this_n_el = self.n_el+0
        self.this_n_nodes = self.n_nodes+0
        self.this_conec = self.conec.copy()
        self.this_coord_nodes = self.coord_nodes.copy()

        if self.is_shape == True:
            self.this_shape_nodes_x = self.shape_nodes_x.copy()
            self.this_shape_nodes_y = self.shape_nodes_y.copy()
            self.this_n_shape_x = self.n_shape_x+0
        if self.is_topology == True:
            self.this_conec = np.array([self.conec[i] for i in range(self.n_el) if self.this_x[i] != 0])
            self.this_n_el = len(self.this_conec)
            if self.this_n_el < self.n_el:
                # print('reduziu o n de elementos')

                self.active_nodes = np.unique(self.this_conec[:,1:])
                # print(self.this_conec)
                # print(self.active_nodes)
                self.this_n_nodes = len(self.active_nodes)
                if self.this_n_nodes < self.n_nodes:
                    # print('reduziu o n de nós')
                    for i in np.unique(np.append(self.forces[:, 0], self.GDL_rest[:, 0])):
                        if i not in self.active_nodes:
                            penalization += self.penal_topo
                    self.this_coord_nodes = np.array([i for i in self.coord_nodes if i[0] in self.active_nodes])
                    # print(self.this_coord_nodes)
                    if self.is_shape == True:
                        for i in range(len(self.shape_nodes_y)-1, -1, -1):
                            for j in range(len(self.shape_nodes_y[i, 0])-1, -1, -1):
                                if self.shape_nodes_y[i, 0][j] not in self.active_nodes:
                                    listinha = self.this_shape_nodes_y[i, 0].copy()
                                    del listinha[j]
                                    self.this_shape_nodes_y[i, 0] = listinha
                            if self.this_shape_nodes_y[i, 0] == []:
                                self.this_shape_nodes_y = np.delete(self.this_shape_nodes_y, i, 0)
                                self.this_x = np.delete(self.this_x, i+self.n_el+self.n_shape_x)
                        if self.n_shape_x != 0:
                            for i in range(len(self.shape_nodes_x)-1, -1, -1):
                                for j in range(len(self.shape_nodes_x[i, 0])-1, -1, -1):
                                    if self.shape_nodes_x[i, 0][j] not in self.active_nodes:
                                        listinha = self.this_shape_nodes_x[i, 0].copy()
                                        del listinha[j]
                                        self.this_shape_nodes_x[i, 0] = listinha
                                if self.this_shape_nodes_x[i, 0] == []:
                                    self.this_shape_nodes_x = np.delete(self.this_shape_nodes_x, i, 0)
                                    self.this_x = np.delete(self.this_x, i+self.n_el)

                            self.this_n_shape_x = len(self.this_shape_nodes_x)
                        else:
                            self.this_n_shape_x = self.n_shape_x
                self.this_x = np.delete(self.this_x, [i for i in range(self.n_el) if self.this_x[i] == 0])
        self.GDL = 2*self.this_n_nodes
        return penalization

    def calc_lenght(self):
        if self.is_shape == True:
            self.x_shape = self.this_x[self.this_n_el:]
            # print(self.this_coord_nodes)
            self.x_size = self.this_x[:self.this_n_el]
            for h in range(len(self.x_shape)):
                if h > self.this_n_shape_x-1:
                    for vs in self.this_shape_nodes_y[h-self.this_n_shape_x,0]:
                        where = np.where(self.this_coord_nodes[:, 0] == vs)[0][0]
                        self.this_coord_nodes[where, 2] = self.x_shape[h]
                else:
                    if self.is_simetric == True:
                        e1 = self.this_shape_nodes_x[h,0][0]
                        e2 = self.this_shape_nodes_x[h,0][1]
                        where1 = np.where(self.this_coord_nodes[:, 0] == e1)[0][0]
                        where2 = np.where(self.this_coord_nodes[:, 0] == e2)[0][0]
                        self.this_coord_nodes[where1, 1] = self.this_coord_nodes[where1, 1] + self.x_shape[h]
                        self.this_coord_nodes[where2, 1] = self.this_coord_nodes[where2, 1] - self.x_shape[h]
                    else:
                        for vs in self.this_shape_nodes_x[h,0]:
                            where = np.where(self.this_coord_nodes[:, 0] == vs)[0][0]
                            self.this_coord_nodes[where, 1] = self.x_shape[h]
        else:
            self.x_size = self.this_x
        self.lenght = np.zeros(self.this_n_el)
        self.cs = np.zeros(self.this_n_el)
        self.sn = np.zeros(self.this_n_el)
        for el in range(self.this_n_el):
            node1 = self.this_conec[el, 1]
            node2 = self.this_conec[el, 2]
            where_node1 = np.where(self.this_coord_nodes[:, 0] == node1)[0][0]
            where_node2 = np.where(self.this_coord_nodes[:, 0] == node2)[0][0]
            sum_x = self.this_coord_nodes[where_node2, 1] - self.this_coord_nodes[where_node1, 1]
            sum_y = self.this_coord_nodes[where_node2, 2] - self.this_coord_nodes[where_node1, 2]
            self.lenght[el] = sqrt(sum_x**2 + sum_y**2)
            self.cs[el] = sum_x/self.lenght[el]
            self.sn[el] = sum_y/self.lenght[el]
    def weight(self):
        self.weight_el = np.zeros(self.this_n_el)
        for el in range(self.this_n_el):
            self.weight_el[el] = self.x_size[el]*self.lenght[el]*self.density

    def rigidez(self):
        K = np.zeros((self.GDL, self.GDL))
        #Calculando a matriz rigidez local de cada elemento
        for el in range(self.this_n_el):

            #Cálculo do comprimento do elemento
            L = self.lenght[el]

            #Propriedades dos elementos
            A = self.x_size[el]
            E = self.E

            #Cossenos e senos diretores do elemento
            cs = self.cs[el]
            sn = self.sn[el]

            #Matriz rigidez local do elemento
            k = E*A/L
            ke = np.array([ [k, -k],
                            [-k, k]])
            T = np.array([  [cs, sn, 0, 0],
                            [0, 0, cs, sn]])
            kg = np.transpose(T).dot(ke).dot(T)

            #Definindo Nodes do elemento atual
            node1 = self.this_conec[el, 1]
            node2 = self.this_conec[el, 2]
            where_node1 = np.where(self.this_coord_nodes[:, 0] == node1)[0][0] + 1
            where_node2 = np.where(self.this_coord_nodes[:, 0] == node2)[0][0] + 1
            #Inserção na matriz rigidez global
            for i in range(2):
                ig = (where_node1-1)*2+i
                for j in range(2):
                    jg = (where_node1 -1)*2+j
                    K[ig, jg] += kg[i, j]

            for i in range(2):
                ig = (where_node2-1)*2+i
                for j in range(2):
                    jg = (where_node2-1)*2+j
                    K[ig, jg] += kg[i+2, j+2]

            for i in range(2):
                ig = (where_node1-1)*2+i
                for j in range(2):
                    jg = (where_node2-1)*2+j
                    K[ig, jg] += kg[i, j+2]
                    K[jg, ig] += kg[j+2, i]

        self.K = K.copy()
        self.Kg = K.copy()

    def calc_forces(self):
        F = np.zeros((self.GDL, 1))
        for i in range(self.n_forces):
            where_i = np.where(self.this_coord_nodes[:, 0] == self.forces[i,0])[0][0]+1
            F[int(2*where_i-2)] = self.forces[i, 1]
            F[int(2*where_i-1)] = self.forces[i, 2]
        self.F = F.copy()
        self.Fg = F.copy()

    def restrics(self):
        for k in range(self.n_rest):
            #Verificando restrições em x
            if self.GDL_rest[k, 1] == 1:
                where_GDL = np.where(self.this_coord_nodes[:, 0] == self.GDL_rest[k, 0])[0][0]+1
                j = 2*where_GDL-2
                #zerando a linha correspondente ao grau de liberdade do no restringido
                for i in range(self.GDL):
                    self.Kg[j,i]=0;  #zerar a linha
                    self.Kg[i,j]=0;  #zerar a coluna
                self.Kg[j,j]=1 #Pois a diagonal principal precisa ter o valor de 1
                self.Fg[j] = 0 #zera a força na direção e nó restringido

        for k in range(self.n_rest):
            #Verificando restrições em y
            if self.GDL_rest[k, 2] == 1:
                where_GDL = np.where(self.this_coord_nodes[:, 0] == self.GDL_rest[k, 0])[0][0]+1
                j = 2*where_GDL-1
                #zerando a linha correspondente ao grau de liberdade do no restringido
                for i in range(self.GDL):
                    self.Kg[j,i]=0;  #zerar a linha
                    self.Kg[i,j]=0;  #zerar a coluna
                self.Kg[j,j]=1 #Pois a diagonal principal precisa ter o valor de 1
                self.Fg[j] = 0 #zera a força na direção e nó restringido

    def desloc_reactions(self):
        #calculo dos deslocamentos
        self.desloc = np.linalg.inv(self.Kg).dot(self.Fg)
        self.reactions = self.K*self.desloc

    def stresses(self):
        self.fn = np.zeros(self.this_n_el)
        self.ten = np.zeros(self.this_n_el)

        for el in range(self.this_n_el):
            node1 = self.this_conec[el, 1]
            node2 = self.this_conec[el, 2]
            where_node1 = np.where(self.this_coord_nodes[:, 0] == node1)[0][0] + 1
            where_node2 = np.where(self.this_coord_nodes[:, 0] == node2)[0][0] + 1
            #comprimento de cada elementos
            L = self.lenght[el]
            #Propriedades
            A = self.x_size[el]
            E = self.E
            #cossenos e senos diretores
            cs = self.cs[el]
            sn = self.sn[el]
            #adicionando variaveis dos deslocamentos dos nos de el
            u1 = self.desloc[where_node1*2-2]
            u2 = self.desloc[where_node2*2-2]
            v1 = self.desloc[where_node1*2-1]
            v2 = self.desloc[where_node2*2-1]
            #constante de rigidez de el
            k=E*A/L;
            #forca e tensao atuante em el
            self.fn[el] = k*(-(u1-u2)*cs-(v1-v2)*sn) #calculo da forca normal
            self.ten[el] = self.fn[el]/A #Tensao normal do elemento
            #calculo das reacoes
            self.reacoes = self.K*self.desloc;

    def convert_to_discrete(self):
        if self.is_discrete == True:
            for el in range(self.this_n_el):
                self.x_size[el] = self.bars_list[round(self.x_size[el])-1]
        else: pass

    def print_results(self):
        try:
            self.desloc
            print('========================================================')
            print('    N             UX                UY')
            print('--------------------------------------------------------')
            for i in range(1, self.this_n_nodes+1):
               print('    ',int(self.this_coord_nodes[i-1,0]),'            ',self.desloc[2*i-2],'            ',self.desloc[2*i-1])
            print('--------------------------------------------------------')
            print('ELEMENTO         FOR�A AXIAL         TENS�O NORMAL')
            print('--------------------------------------------------------')
            for i in range(self.this_n_el):
               print('    ',self.this_conec[i,0],'            ',self.fn[i],'            ',self.ten[i])
            print('========================================================')
        except:
            print('========================================================')
            print('================= ESTRUTURA É INSTÁVEL =================')
            print('========================================================')

    def plot_truss(self):
        y_major = 0
        y_minor = 10e2
        if self.is_shape == True:
            for el in range(self.n_el):
                xi, yi = self.coord_nodes[self.conec[el, 1]-1,1], self.coord_nodes[self.conec[el, 1]-1,2]
                xf, yf = self.coord_nodes[self.conec[el, 2]-1,1], self.coord_nodes[self.conec[el, 2]-1,2]
                plt.plot([xi, xf], [yi, yf], c='cyan')
        print(self.this_coord_nodes)
        for el in range(self.this_n_el):
            where_node1 = np.where(self.this_coord_nodes[:, 0] == self.this_conec[el, 1])[0][0]
            where_node2 = np.where(self.this_coord_nodes[:, 0] == self.this_conec[el, 2])[0][0]
            xi, yi = self.this_coord_nodes[where_node1,1], self.this_coord_nodes[where_node1,2]
            xf, yf = self.this_coord_nodes[where_node2,1], self.this_coord_nodes[where_node2,2]
            plt.plot([xi, xf], [yi, yf], c='black')

            if yf > y_major:
                y_major = yf
            elif yf < y_minor:
                y_minor = yf
            if yi > y_major:
                y_major = yi
            elif yi < y_minor:
                y_minor = yi

        graph_size = y_major - y_minor
        for rest in self.GDL_rest:
            where_node_GDL = np.where(self.this_coord_nodes[:, 0] == rest[0])[0][0]
            xd, yd = self.this_coord_nodes[where_node_GDL, 1], self.this_coord_nodes[where_node_GDL, 2]
            plt.plot(xd, yd-graph_size*0.013, c='black', marker='^', markersize=round(graph_size*1.2))
        plt.show()


    def f_objective(self, k):
        weight_this = 0
        self.cost += 1
        self.x = k.copy()
        penalization = 0
        penalization = self.topology_correction()
        weight_this += penalization
        if penalization == 0:
            self.calc_lenght()
            self.rigidez()
            self.calc_forces()
            self.restrics()
            if self.is_topology == True and np.linalg.det(self.Kg) <= 0: #1e-7
                weight_this += self.penal_topo
            else:
                self.desloc_reactions()
                self.stresses()
                self.weight()
                if self.is_buckling == True:
                    for el in range(self.this_n_el):
                        buck_value = self.buckling(self.lenght[el], self.E, self.x_size[el])
                        if -self.ten[el]/buck_value -1 > 0:
                            # print(f'EL {el+1} não passou em buckling')
                            # self.weight_el[el] =+ self.penal_others
                            # self.weight_el[el] =+ max((-self.ten[el]/buck_value -1)*self.penal_others, 20, self.penal_others*0.01)
                            # self.weight_el[el] =+ (-self.ten[el]/buck_value -1)*self.penal_others
                            self.weight_el[el] =+ (1+(-self.ten[el]/buck_value -1))*self.penal_others
                for el in range(self.this_n_el):
                    if abs(self.ten[el])/self.ten_adm - 1 > 0:
                        # print(f'EL {el+1} não passou em tensao')
                        # self.weight_el[el] =+ self.penal_others
                        # self.weight_el[el] =+ max((abs(self.ten[el])/self.ten_adm - 1)*self.penal_others, 20, self.penal_others*0.01)
                        # self.weight_el[el] =+ (abs(self.ten[el])/self.ten_adm - 1)*self.penal_others #(Tensao atuante / admissível) - 1 > 0 (falha) #(deslocamento também)
                        self.weight_el[el] =+ (1+(abs(self.ten[el])/self.ten_adm - 1))*self.penal_others
                for no in range(self.this_n_nodes):
                    # if abs(self.desloc[2*(no+1)-2])/self.max_deflection - 1 > 0 or abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1 > 0:
                    if abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1 > 0:
                        # print(f'NO {no+1} não passou em deslocamento')
                        # weight_this += self.penal_others
                        # weight_this += max((abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1)*self.penal_others, 20, self.penal_others*0.01)
                        # weight_this += (abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1)*self.penal_others
                        weight_this += (1+(abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1))*self.penal_others
                weight_this += sum(self.weight_el)
        return(weight_this)

    def evaluate(self, x):
        #self.weight_total = np.zeros((x.shape[0],1))
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.f_objective)(b) for b in x)

        weight_total = np.zeros((x.shape[0],1))

        for k in range(len(x)):
            weight_total[k, 0] = results[k]

        return weight_total

    # def evaluate(self, x):
    #     self.iterator = x.shape[0]
    #     weight_total = np.zeros((self.iterator,1))
    #     for k in range(self.iterator):
    #         self.cost += 1
    #         self.x = x[k].copy()
    #         penalization = 0
    #         penalization = self.topology_correction()
    #         weight_total[k] += penalization
    #         if penalization == 0:
    #             self.calc_lenght()
    #             # self.convert_to_discrete()
    #             # print(self.x_size)
    #             # print(self.this_coord_nodes)
    #             self.rigidez()
    #             self.calc_forces()
    #             self.restrics()
    #             if self.is_topology == True and np.linalg.det(self.Kg) <= 0: #1e-7
    #                 weight_total[k] += self.penal_topo
    #             else:
    #                 self.desloc_reactions()
    #                 self.stresses()
    #                 self.weight()
    #                 if self.is_buckling == True:
    #                     for el in range(self.this_n_el):
    #                         buck_value = self.buckling(self.lenght[el], self.E, self.x_size[el])
    #                         if self.ten[el]/buck_value +1 < 0:
    #                             # print(f'EL {el+1} não passou em buckling')
    #                             self.weight_el[el] =+ (abs(self.ten[el])-buck_value)*self.penal_others
    #                 for el in range(self.this_n_el):
    #                     if abs(self.ten[el])/self.ten_adm - 1 > 0:
    #                         # print(f'EL {el+1} não passou em tensao')
    #                         self.weight_el[el] =+ (abs(self.ten[el])-self.ten_adm)*self.penal_others #(Tensao atuante / admissível) - 1 > 0 (falha) #(deslocamento também)
    #                 for no in range(self.this_n_nodes):
    #                     # if abs(self.desloc[2*(no+1)-2])/self.max_deflection - 1 > 0 or abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1 > 0:
    #                     if abs(self.desloc[2*(no+1)-1])/self.max_deflection - 1 > 0:
    #                         # print(f'NO {no+1} não passou em deslocamento')
    #                         weight_total[k] += self.penal_others
    #                 weight_total[k] += sum(self.weight_el)
    #     return weight_total

    def evaluate_solo(self, x):
        results = [self.f_objective(x[0])]
        weight_total = np.zeros((x.shape[0],1))

        for k in range(len(x)):
            weight_total[k, 0] = results[k]

        return weight_total


if __name__=='__main__':
    from geometry import Geometry
    from plot_truss import plot_truss
    struc_number = 6
    geo = Geometry(struc_number=struc_number)
    struc = Structure(geo)

    # x = np.array([[2.289671781034572, 28.773476616199467, 16.569363554851478, 22.75161005957987, 32.28781135540524, 28.320441817391238, 10.66860222292101, 3.3982233904905508, 30.793279821741887, 6.153907754396527, 1.3282651995986956, 13.815865882468573, 4.572, 11.081040121261468]])
    # x = np.array([[0.323, 5.152, 0, 0, 9.678, 0, 0, 1.3, 0, 7.285, 5.516, 0, 0, 2.907, 0, 1.399, 0, 0, 1.251, 0, 5.201]])*10**-4
    # x = np.array([[18.1192847727191, 16.29931196119466, 11.730720892812496, 21.268070993658377, 19.104712603858918, 16.06483303423725, 3.5723718612911743, 15.302532706444103, 14.895440167592923, 14.027923471545998, 19.40213621376308, 12.82852305513544, 19.201634817548978, 13.13744642358602, 18.335591059542434, 3.1558542937472285, 5.881606343456308, 3.226464379549929, 3.319199021888311, 1.9363419126773012, 0.20940566570552274, 0.22229805535764183, 1.1587969999184513]])
    # x = np.array([[16.0, 16.0, 16.0, 22.0, 19.0, 8.0, 11.0, 15.0, 14.0, 14.0, 1.0, 11.0, 1.0, 1.0, 21.0, 3.2241109999999997, 5.615408, 2.54, 2.54, 1.3999690000000002, -0.094016, 0.508, 1.387773]])
    x = np.array([[15.920578755939152, 16.442575190262765, 14.132109044312545, 20.710715128081013, 19.91753869395259, 7.332602790032157, 5.1320943741626435, 12.040499353906277, 28.23442982714642, 13.962369234840857, 5.910058754644449, 11.157607423662851, 21.692506867323928, 3.531987550989872, 16.97143941180167, 2.7747707415301894, 6.3996359529169515, 2.9399404163749545, 3.3935732152670623, 1.4323339007405003, 0.14429669994114266, -0.2917584612245625, 1.415860628650555]])
    x = np.array([[2.25210120387059, 20.272004865166025, 7.283378939765933, 17.67983702179543, 20.942419479093033, 15.139484196652464, 3.7494907292596755, 1.2356866567461753, 20.992682168042126, 2.1323081841848888, 2.290954406233513, 19.11508449091374, 11.774331488634097, 9.169504060810073]])
    x = np.array([[11.989079540656897, 12.362375492173292, 5.386240404379422, 14.607363173452413, 13.956835922245446, 11.197376013359936, 3.1318816999423404, 5.2395207622723206, 15.37524330413551, 2.3797298716434234, 5.309859950430858, 2.9955050060964483, 14.597092371627093, 6.749670947669942, 10.262208793137003, 2.8137151258593858, 6.005631060901536, 2.673776759348127, 2.8117126552561027, 1.4177878308583887, 0.4967054627290282, 0.3162077577488979, 1.3840769858156659]])
    x = np.array([[11.989661925573868, 10.01798137729716, 6.380654007842207, 15.154542561057477, 13.896571783672233, 9.881125487221764, 4.202005208773968, 4.063699844119029, 6.031613318415569, 5.328618202415982, 1.1635526441328155, 4.850193132961671, 14.98572528630728, 6.935682055964732, 9.767512379187725, 2.8204974287353632, 6.077120866359563, 2.6832669684458637, 2.880560425057004, 1.7390773169361327, 0.48467700693489646, 0.19956043039523702, 1.1816136184023152]])
    x = np.array([[3.900441339457766e-05, 0.0004891140731308391, -0.0001276345727518114, 0.00038945876448709025, 0.000968728125635872, -6.713877576350421e-05, -5.299189118462616e-05, 0.0001611877018105287, -6.202770262145969e-05, 0.0006851342572575726, 0.0006839931779108156, -7.659729293605962e-05, -4.264394610848403e-05, 0.00036068861867401297, -3.652506361872006e-05, -7.154753199379427e-05, -6.692225952422319e-05, -6.521289754178693e-05, 3.6379910878792184e-05, -9.465498552811241e-05, 0.00025587099167578424]])
    x = np.array([[0.00045163972876295844, 0.0007131471347557671, 0.00019109038891729473, 0.0006455239739792649, -3.712234094988117e-05, 0.0004224577428044203, 0.0014381440232653735, -0.00010124055440539997, 0.00034024758379243243, 0.0002983860880954788, 0.001275775844529781, -5.8979814650486533e-05, 4.796575336840761e-05, -0.00011529196056711548, 0.0007512730967021465, -4.961743470520407e-05, 9.670372886058748e-05, -0.00012109508142540275, 0.0006122108478956272, 6.78461805578066e-05, 1.3694744963815035e-05, 1.5673368824853628, -1.3154481242791543, -1.460620697512406, 3.455098657289911, 3.463735490028123, 3.5276433739806405, 5.5471601654914355] ])
    y = struc.evaluate_solo(x)
    print(f'y = {y[0][0]}')
    print(x)
    struc.print_results()
    plot_truss(np.array([x[0].tolist()]), struc_number)
    # print(max(struc.desloc)[0])

