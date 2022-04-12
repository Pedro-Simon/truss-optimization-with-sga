import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = os.path.abspath('..')
sys.path.append(path)

if __name__ == "__main__":
    from structure import Structure
    from geometry import Geometry
else:
    from pysga.structure import Structure
    from pysga import Geometry

def plot_truss(x, struc_number=None, path_name=None, has_title=True, has_axis=True, is_horizontal=False, inside_text=False, y_space = 0.2):

    geo = Geometry(struc_number)
    struc = Structure(geo)
    y = struc.evaluate_solo(x)

    f = plt.figure(figsize=(9, 6))
    ax = f.add_subplot()

    if geo.is_shape == True:
        for el in range(geo.n_el):
            xi, yi = geo.coord_nodes[geo.conec[el, 1]-1,1], geo.coord_nodes[geo.conec[el, 1]-1,2]
            xf, yf = geo.coord_nodes[geo.conec[el, 2]-1,1], geo.coord_nodes[geo.conec[el, 2]-1,2]
            ax.plot([xi, xf], [yi, yf], c='cyan')

    for el in range(struc.this_n_el):
        where_node1 = np.where(struc.this_coord_nodes[:, 0] == struc.this_conec[el, 1])[0][0]
        where_node2 = np.where(struc.this_coord_nodes[:, 0] == struc.this_conec[el, 2])[0][0]
        xi, yi = struc.this_coord_nodes[where_node1,1], struc.this_coord_nodes[where_node1,2]
        xf, yf = struc.this_coord_nodes[where_node2,1], struc.this_coord_nodes[where_node2,2]
        ax.plot([xi, xf], [yi, yf], c='black')

        vector = np.array([xf - xi, yf - yi])
        middle = np.array([(xf - xi)/2+xi , (yf - yi)/2+yi])

        if inside_text:
            y_space = 0

        if vector[0] != 0:
            text_position = np.array([min(((-vector[1]*y_space)/vector[0]),y_space),y_space]) + middle
        else:
            text_position = np.array([y_space,y_space]) + middle
        ax.text(text_position[0], text_position[1], str(struc.this_conec[el, 0]), backgroundcolor='white', bbox=dict(boxstyle="Circle, pad=0.0", color='white'))

    for rest in struc.GDL_rest:
        where_node_GDL = np.where(struc.this_coord_nodes[:, 0] == rest[0])[0][0]
        xd, yd = struc.this_coord_nodes[where_node_GDL, 1], struc.this_coord_nodes[where_node_GDL, 2]
        if is_horizontal:
            ax.plot(xd-0.10, yd+0.01, c='black', marker='>', markersize=round(8))
        else:
            ax.plot(xd-0.03, yd-0.1, c='black', marker='^', markersize=round(8))

    if has_axis:
        x_tick = np.unique(struc.this_coord_nodes[:, 1])
        x_tick = np.sort(x_tick)
        y_tick = np.unique(struc.this_coord_nodes[:, 2])
        y_tick = np.sort(y_tick)
        plt.xticks(x_tick)
        plt.yticks(y_tick)

        for direction in ["right", "top"]:
        # hides borders
            ax.spines[direction].set_visible(False)
    else:
        for direction in ["right", "top", "bottom", "left"]:
        # hides borders
            ax.spines[direction].set_visible(False)
        plt.xticks([])
        plt.yticks([])

    if has_title:
        plt.title('Estrutura nº'+str(geo.struc_number))
    plt.text(0.95, 0.95,'Massa: '+str(round(y[0][0],2))+' kg', ha='right', va='top', transform=ax.transAxes)
    ax.set_aspect('equal', 'box')

    if path_name == None:
        f.savefig(path+'/demo.png', bbox_inches='tight')
    else:
        f.savefig(path_name, bbox_inches='tight')

if __name__ == '__main__':
    x = np.array([[2.997994838676734, 21.987503904112486, 16.901665973371266, 8.84349709681765, 24.76650744687407, 21.265147103556995, 7.407767384299452, 0.8815099568304301, 24.054253261990898, 0.3609949223957451, 0.6753692851359133, 13.588121030522753, 4.717637446400265, 4.757491579704007]])
    plot_truss(x, 4)
