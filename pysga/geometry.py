import numpy as np

class Geometry:
    def __init__(self, struc_number=1):
        #coordenadas de cada no: [n_node, coord_x, coord_y]
        self.struc_number = struc_number

        if struc_number == 1:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,  0],
                                        [2,  5,  0],
                                        [3,  5,  5*np.tan(np.radians(60))],
                                        [4,  10, 5*np.tan(np.radians(60))]])
                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   2],
                                    [2,   3,   4],
                                    [3,   1,   3],
                                    [4,   2,   4],
                                    [5,   2,   3]])
                #FORÇAS ATUANTES
            self.forces = np.array([[4,   np.cos(np.radians(45))*400e3,   -np.sin(np.radians(45))*400e3]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [2,   0,   1]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = False
            self.is_shape = False
            self.is_simetric = False
            self.bars_list = [(i+1)*100e-6 for i in range(150)] #SE FOR DISCRETA, QUAL A LISTA DE SIZES

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 15000e-6 #em metros
            min_area = 10e-6 #em metros

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[[2], 0.7,  -0.7], #[NÓS, LIM SUP, LIM INF]
                                           [[3], 0.7,  -0.7],
                                           [[4], 0.7,  -0.7],
                                           [[6], 0.7,  -0.7],
                                           [[7], 0.7,  -0.7],
                                           [[8], 0.7,  -0.7]])

                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[2], 0.7,  -0.7], #[NÓS, LIM SUP, LIM INF]
                                           [[3], 0.7,  -0.7],
                                           [[4], 0.7,  -0.7],
                                           [[6], 0.7,  -0.7],
                                           [[7], 0.7,  -0.7],
                                           [[8], 0.7,  -0.7]])
            #CASO NÃO TENHA SHAPE, DEIXAR VALOR QUALQUER E DAR NONE EM DESLOC_X E DESLOC_Y (MAX E MIN)

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 200e9
            self.ten_adm = 220e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 0.02 #metros
            self.density = 7850 #kg/m3

        elif struc_number == 2:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,  3.048],
                                        [2,  3.048,  3.048],
                                        [3,  6.096,  3.048],
                                        [4,  9.144,  3.048],
                                        [5,  0,  0],
                                        [6,  3.048,  0],
                                        [7,  6.096,  0],
                                        [8,  9.144, 0]])
                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   2],
                                    [2,   2,   3],
                                    [3,   3,   4],
                                    [4,   5,   6],
                                    [5,   6,   7],
                                    [6,   7,   8],
                                    [7,   2,   6],
                                    [8,   3,   7],
                                    [9,   4,   8],
                                    [10,   1,   6],
                                    [11,   2,   5],
                                    [12,   2,   7],
                                    [13,   3,   6],
                                    [14,   3,   8],
                                    [15,   4,   7]])
                #FORÇAS ATUANTES
            self.forces = np.array([[8,   0,   -44482.2161]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [5,   1,   1]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = True
            self.bars_list = np.array([0.716, 0.910, 1.123, 1.419, 1.742, 1.852,
                              2.239, 2.839, 3.478, 6.155, 6.974, 7.574, 8.600,
                              9.600, 11.381, 13.819, 17.400,18.064, 20.200,
                              23.000, 24.600, 31.000, 38.400, 42.400, 46.400,
                              55.000, 60.000, 70.000, 86.000, 92.193, 110.774,
                              123.742])*1e-4 #SE FOR DISCRETA, QUAL A LISTA DE SIZES

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 32.4 #em metros
            min_area = 0.6 #em metros

            self.is_shape = True #TRUE caso tenha otimização shape // FALSE caso não tenha
            self.is_simetric = False

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[[2,6], 3.556, 2.54], #[NÓS, LIM SUP, LIM INF]
                                           [[3,7], 6.604,  5.588]],dtype=object)

                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[2], 3.556,  2.54], #[NÓS, LIM SUP, LIM INF]
                                           [[3], 5.588,  2.54],
                                           [[4], 2.286,  1.27],
                                           [[6], 0.508,  -0.508],
                                           [[7], 0.508,  -0.508],
                                           [[8], 1.524,  0.508]], dtype=object)

            self.is_topology = False #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #Define quais elementos podem ser retirados em topology

            self.is_buckling = True #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.591e6
            self.ten_adm = 172.369e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 20000 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        elif struc_number == 3:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,  3.048],
                                        [2,  3.048,  3.048],
                                        [3,  6.096,  3.048],
                                        [4,  9.144,  3.048],
                                        [5,  0,  0],
                                        [6,  3.048,  0],
                                        [7,  6.096,  0],
                                        [8,  9.144, 0]])
                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   2],
                                    [2,   2,   3],
                                    [3,   3,   4],
                                    [4,   5,   6],
                                    [5,   6,   7],
                                    [6,   7,   8],
                                    [7,   2,   6],
                                    [8,   3,   7],
                                    [9,   4,   8],
                                    [10,   1,   6],
                                    [11,   2,   5],
                                    [12,   2,   7],
                                    [13,   3,   6],
                                    [14,   3,   8],
                                    [15,   4,   7]])
                #FORÇAS ATUANTES
            self.forces = np.array([[8,   0,   -44482.2161]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [5,   1,   1]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = True
            self.bars_list = np.array([0, 0, 0, 0, 0, 0, 0.716, 0.910, 1.123, 1.419, 1.742, 1.852,
                              2.239, 2.839, 3.478, 6.155, 6.974, 7.574, 8.600,
                              9.600, 11.381, 13.819, 17.400,18.064, 20.200,
                              23.000, 24.600, 31.000, 38.400, 42.400, 46.400,
                              55.000, 60.000, 70.000, 86.000, 92.193, 110.774,
                              123.742])*1e-4 #SE FOR DISCRETA, QUAL A LISTA DE SIZES

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 38 #em metros
            min_area = 1 #em metros

            self.is_shape = True #TRUE caso tenha otimização shape // FALSE caso não tenha
            self.is_simetric = False

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[[2,6], 3.556, 2.54], #[NÓS, LIM SUP, LIM INF]
                                           [[3,7], 6.604,  5.588]],dtype=object)

                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[2], 3.556,  2.54], #[NÓS, LIM SUP, LIM INF]
                                           [[3], 5.588,  2.54],
                                           [[4], 2.286,  1.27],
                                           [[6], 0.508,  -0.508],
                                           [[7], 0.508,  -0.508],
                                           [[8], 1.524,  0.508]], dtype=object)

            self.is_topology = True #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #Define quais elementos podem ser retirados em topology

            self.is_buckling = True #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.591e6
            self.ten_adm = 172.369e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 20000 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        elif struc_number == 4:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,   9.144],
                                        [2,  9.144,   9.144],
                                        [3,  18.288,   9.144],
                                        [4,  0,  0],
                                        [5,  9.144,  0],
                                        [6,  18.288,  0]])
                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   4],
                                    [2,   4,   5],
                                    [3,   2,   4],
                                    [4,   1,   5],
                                    [5,   1,   2],
                                    [6,   5,   6],
                                    [7,   2,   5],
                                    [8,   3,   5],
                                    [9,   2,   6],
                                    [10,   3,   6],
                                    [11,   2,   3]])
                #FORÇAS ATUANTES
            # self.forces = np.array([[5,   0,   -453592.37],
            #                         [6,   0,   -453592.37]])
            self.forces = np.array([[5,   0,   -444822.16152605],
                                    [6,   0,   -444822.16152605]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [4,   1,   1]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = True
            # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            self.bars_list = np.array([0, 0, 0, 0, 0, 0, 0.001045159, 0.001161288,0.001535481,0.001690319,
            0.001858061,0.001993544,0.002019351,0.002180641,0.002341931,0.002477414,0.002496769,
            0.002696769,0.002896768,0.003096768,0.003206445,0.003303219,0.003703218,0.004658055,
            0.005141925,0.00741934,0.00870966,0.008967724,0.009161272,0.00999998,0.01032256,
            0.012129008,0.012838684, 0.01419352, 0.014774164, 0.01709674,0.0193548,0.02161286]) #SE FOR DISCRETA, QUAL A LISTA DE SIZES

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 38 #em metros
            min_area = 7 #em metros

            self.is_shape = True #TRUE caso tenha otimização shape // FALSE caso não tenha
            self.is_simetric = False

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[]],dtype=object)


                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[1], 25.40,  4.572], #[NÓS, LIM SUP, LIM INF]
                                           [[2], 25.40,  4.572],
                                           [[3], 25.40,  4.572]], dtype=object)
            self.is_topology = True #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 3, 4, 5, 7, 8, 9, 10, 11] #Define quais elementos podem ser retirados em topology

            self.is_buckling = False #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.5909e6
            self.ten_adm = 172.369e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 50.8e-3 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        elif struc_number == 5:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,  0],
                                        [2,  3.048,  0],
                                        [3,  6.096,  0],
                                        [4,  9.144,  0],
                                        [5,  12.192,  0],
                                        [6,  0,  3.048],
                                        [7,  3.048,  3.048],
                                        [8,  9.144,  3.048],
                                        [9,  12.192,  3.048],
                                        [10,  3.048,  6.096],
                                        [11,  6.096,  6.096],
                                        [12,  9.144,  6.096]])

                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   2],
                                     [2,   2,   3],
                                     [3,   6,   7],
                                     [4,   10,   11],
                                     [5,   1,   6],
                                     [6,   2,   7],
                                     [7,   7,   10],
                                     [8,   2,   10],
                                     [9,   1,   7],
                                     [10,   2,   6],
                                     [11,   6,   10],
                                     [12,   3,   6],
                                     [13,   3,   7],
                                     [14,   3,   10],
                                     [15,   1,   10],
                                     [16,   6,   11],
                                     [17,   7,   11],
                                     [18,   2,   11],
                                     [19,   3,   11],
                                     [20,   7,   8],
                                     [21,   10,   12],
                                     [22,   4,   5],
                                     [23,   3,   4],
                                     [24,   8,   9],
                                     [25,   11,   12],
                                     [26,   5,   9],
                                     [27,   4,   8],
                                     [28,   8,   12],
                                     [29,   4,   12],
                                     [30,   5,   8],
                                     [31,   4,   9],
                                     [32,   9,   12],
                                     [33,   3,   9],
                                     [34,   3,   8],
                                     [35,   3,   12],
                                     [36,   5,   12],
                                     [37,   9,   11],
                                     [38,   8,   11],
                                     [39,   4,   11]])
                #FORÇAS ATUANTES
            self.forces = np.array([[2,   0,   -9071.847*9.8],
                                    [3,   0,   -9071.847*9.8],
                                    [4,   0,   -9071.847*9.8]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [5,   0,   1]])

            self.is_simetric = True

            self.size_vars = np.array([[1, 22], [2, 23], [3, 24], [4, 25], [5, 26], [6, 27], [7, 28], [8, 29], [9, 30], [10, 31],
            [11, 32], [12, 33], [13, 34], [14, 35], [15, 36], [16, 37], [17, 38], [18, 39], [19], [20], [21]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = False

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 0.001452 #em metros 2
            min_area = -0.000145 #em metros 2

            self.is_shape = False #TRUE caso tenha otimização shape // FALSE caso não tenha

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[]],dtype=object)


                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[1], 25.40,  4.572], #[NÓS, LIM SUP, LIM INF]
                                           [[2], 25.40,  4.572],
                                           [[3], 25.40,  4.572]], dtype=object)
            self.is_topology = True #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33 ,34, 35, 36, 37, 38, 39] #Define quais elementos podem ser retirados em topology

            self.min_topology = 3.2258e-5 #metros quadrados

            self.is_buckling = False #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.591e6
            self.ten_adm = 137.895e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 50.8e-3 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        elif struc_number == 6:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,  0],
                                        [2,  3.048,  0],
                                        [3,  6.096,  0],
                                        [4,  9.144,  0],
                                        [5,  12.192,  0],
                                        [6,  0,  3.048],
                                        [7,  3.048,  3.048],
                                        [8,  9.144,  3.048],
                                        [9,  12.192,  3.048],
                                        [10,  3.048,  6.096],
                                        [11,  6.096,  6.096],
                                        [12,  9.144,  6.096]])

                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   2],
                                     [2,   2,   3],
                                     [3,   6,   7],
                                     [4,   10,   11],
                                     [5,   1,   6],
                                     [6,   2,   7],
                                     [7,   7,   10],
                                     [8,   2,   10],
                                     [9,   1,   7],
                                     [10,   2,   6],
                                     [11,   6,   10],
                                     [12,   3,   6],
                                     [13,   3,   7],
                                     [14,   3,   10],
                                     [15,   1,   10],
                                     [16,   6,   11],
                                     [17,   7,   11],
                                     [18,   2,   11],
                                     [19,   3,   11],
                                     [20,   7,   8],
                                     [21,   10,   12],
                                     [22,   4,   5],
                                     [23,   3,   4],
                                     [24,   8,   9],
                                     [25,   11,   12],
                                     [26,   5,   9],
                                     [27,   4,   8],
                                     [28,   8,   12],
                                     [29,   4,   12],
                                     [30,   5,   8],
                                     [31,   4,   9],
                                     [32,   9,   12],
                                     [33,   3,   9],
                                     [34,   3,   8],
                                     [35,   3,   12],
                                     [36,   5,   12],
                                     [37,   9,   11],
                                     [38,   8,   11],
                                     [39,   4,   11]])
                #FORÇAS ATUANTES
            self.forces = np.array([[2,   0,   -9071.847*9.8],
                                    [3,   0,   -9071.847*9.8],
                                    [4,   0,   -9071.847*9.8]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [5,   0,   1]])

            self.is_simetric = True

            self.size_vars = np.array([[1, 22], [2, 23], [3, 24], [4, 25], [5, 26], [6, 27], [7, 28], [8, 29], [9, 30], [10, 31],
            [11, 32], [12, 33], [13, 34], [14, 35], [15, 36], [16, 37], [17, 38], [18, 39], [19], [20], [21]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = False

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 0.001452 #em metros 2
            min_area = -0.000145 #em metros 2

            self.is_shape = True #TRUE caso tenha otimização shape // FALSE caso não tenha

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[[6,9], 3.047,  -3.047], #[NÓS, LIM SUP, LIM INF]
                                           [[7,8], 3.047,  -3.047],
                                           [[10,12], 3.047,  -3.047]],dtype=object)


                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[6,9], 6.095,  0.001], #[NÓS, LIM SUP, LIM INF]
                                           [[7,8], 6.095,  0.001],
                                           [[10,12], 9.143,  3.049],
                                           [[11], 9.143,  3.049]], dtype=object)
            self.is_topology = True #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33 ,34, 35, 36, 37, 38, 39] #Define quais elementos podem ser retirados em topology

            self.min_topology = 3.2258e-5 #metros quadrados

            self.is_buckling = False #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.591e6
            self.ten_adm = 137.895e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 50.8e-3 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        elif struc_number == 7:
            #DEFININDO A GEOMETRIA DA ESTRUTURA
                #COORDENADAS DOS NÓS
            self.coord_nodes = np.array([[1,  0,   9.144],
                                        [2,  9.144,   9.144],
                                        [3,  18.288,   9.144],
                                        [4,  0,  0],
                                        [5,  9.144,  0],
                                        [6,  18.288,  0]])
                #MATRIZ CONECTIVIDADE
            self.conec = np.array(  [[1,   1,   4],
                                    [2,   4,   5],
                                    [3,   2,   4],
                                    [4,   1,   5],
                                    [5,   1,   2],
                                    [6,   5,   6],
                                    [7,   2,   5],
                                    [8,   3,   5],
                                    [9,   2,   6],
                                    [10,   3,   6],
                                    [11,   2,   3]])
                #FORÇAS ATUANTES
            # self.forces = np.array([[5,   0,   -453592.37],
            #                         [6,   0,   -453592.37]])
            self.forces = np.array([[5,   0,   -444822.16152605],
                                    [6,   0,   -444822.16152605]])

                #GRAUS DE LIBERDADE DOS APOIOS
            self.GDL_rest = np.array(   [[1,   1,   1],
                                        [4,   1,   1]])

                #DEFININDO O TIPO DE VARIÁVEL SIZE (DISCRETA OU CONTÍNUA
            self.is_discrete = True
            # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            self.bars_list = np.array([0, 0, 0, 0, 0, 0, 0.001045159, 0.001161288,0.001535481,0.001690319,
            0.001858061,0.001993544,0.002019351,0.002180641,0.002341931,0.002477414,0.002496769,
            0.002696769,0.002896768,0.003096768,0.003206445,0.003303219,0.003703218,0.004658055,
            0.005141925,0.00741934,0.00870966,0.008967724,0.009161272,0.00999998,0.01032256,
            0.012129008,0.012838684, 0.01419352, 0.014774164, 0.01709674,0.0193548,0.02161286]) #SE FOR DISCRETA, QUAL A LISTA DE SIZES

                #LIMITES SUPERIOR E INFERIOR PARA SIZE
            max_area = 38 #em metros
            min_area = 7 #em metros

            self.is_shape = False #TRUE caso tenha otimização shape // FALSE caso não tenha
            self.is_simetric = False

                #COORDENADAS X --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_x = np.array([[]],dtype=object)


                #COORDENADAS Y --> NÓS E LIMITES SUPERIOR E INFERIOR
            self.shape_nodes_y = np.array([[[1], 25.40,  4.572], #[NÓS, LIM SUP, LIM INF]
                                           [[2], 25.40,  4.572],
                                           [[3], 25.40,  4.572]], dtype=object)
            self.is_topology = True #TRUE caso tenha otimização topology // FALSE caso não tenha
            topology_el = [1, 3, 4, 5, 7, 8, 9, 10, 11] #Define quais elementos podem ser retirados em topology

            self.is_buckling = False #TRUE se tiver buckling e FALSE se não

                #VARIÁVEIS DO MATERIAL
            self.mod_elast = 68947.5909e6
            self.ten_adm = 172.369e6 #220 MPA = 220e6 N/m2
            self.max_deflection = 50.8e-3 #metros #para retirar a restrição
            self.density = 2767.990 #kg/m3

        #DEFININDO VARIÁVEIS COMPLEMENTARES
        self.n_nodes = np.shape(self.coord_nodes[:,0])[0]
        self.n_el = np.shape(self.conec[:,0])[0]
        self.n_forces = np.shape(self.forces[:,0])[0]
        self.n_rest = np.shape(self.GDL_rest[:,0])[0]
        areas_sup = np.ones(self.n_el)*max_area
        areas_inf = np.ones(self.n_el)*min_area

        if self.is_simetric == True:
            areas_sup = np.ones(len(self.size_vars))*max_area
            areas_inf = np.ones(len(self.size_vars))*min_area
        else:
            areas_sup = np.ones(self.n_el)*max_area
            areas_inf = np.ones(self.n_el)*min_area

        if self.is_shape == True:
            shape_sup = np.array([])
            shape_inf = np.array([])
            if self.shape_nodes_x.shape[1] != 0:
                shape_sup = np.append(shape_sup, self.shape_nodes_x[:, 1].copy())
                shape_inf = np.append(shape_inf, self.shape_nodes_x[:, 2].copy())
                self.n_shape_x = len(self.shape_nodes_x[:,1])
            else:
                self.n_shape_x = 0
            if self.shape_nodes_y.shape[1] != 0:
                shape_sup = np.append(shape_sup, self.shape_nodes_y[:, 1].copy())
                shape_inf = np.append(shape_inf, self.shape_nodes_y[:, 2].copy())

            self.lim_sup = np.append(areas_sup, shape_sup).astype('float64')
            self.lim_inf = np.append(areas_inf, shape_inf).astype('float64')
        else:
            self.lim_sup = areas_sup
            self.lim_inf = areas_inf

        if self.is_topology == True:
            if self.is_simetric == True:
                for n in range(len(self.size_vars)):
                    if self.conec[n, 0] in topology_el:
                        self.lim_inf[n] = min(0, self.lim_inf[n])
            else:
                for n in range(self.n_el):
                    if self.conec[n, 0] in topology_el:
                        self.lim_inf[n] = 0


    #DEFININDO A FUNCTION BUCKLING
    def buckling(self, lenght, mod_elast, area):
        return (100*mod_elast*area)/(8*lenght**2)

if __name__ == '__main__':
    geo = Geometry(3)
    print(geo.coord_nodes,
    geo.conec,
    geo.forces,
    geo.GDL_rest,
    geo.lim_sup,
    geo.lim_inf)
