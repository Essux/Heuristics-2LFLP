from math import sqrt, inf
import matplotlib.pyplot as plt
import cProfile, pstats
from objects import Solution

# Retorna la distancia entre dos puntos
def dist(a, b):
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

# Cálculo de la función objetivo
def obj_function(sel_level1, sel_level2):
    func_obj = 0

    for l in sel_level1:
        for cap, cost in zip(l.u, l.c):
            # Suma del flujo entre cada pareja por el costo
            # unitario de transporte
            func_obj += cap * cost

    for l in sel_level2:
        for cap, cost in zip(l.u, l.c):
            func_obj += cap * cost

    return func_obj

# Calcular pesos entre todas las parejas de puntos
def calculateWeights(level1, level2, clients):
    for l in level1:
        for i in range(len(l.c)):
            l.c[i] = dist(l, clients[i])

    for l in level2:
        for i in range(len(l.c)):
            l.c[i] = dist(l, level1[i])

# Graficar un conjunto de instalaciones
def plotAnnot(li, color):
    plt.scatter([x.x for x in li], [x.y for x in li], color=color)
    for l in li:
        plt.annotate(l.i+1, (l.x+0.5, l.y+0.5))

# Graficar flujo entre instalaciones
def plotFlow(li1, li2, color, mxflow):
    for l in li1:
        for c in li2:
            f = l.u[c.i]
            if f > 0: plt.plot([l.x, c.x], [l.y, c.y], color = color, linewidth=f/mxflow*3)

# Graficar una solución
def plotSolution(level1, level2, clients):
    # Graficar solución
    mxflow = max(max([max(x.u) for x in level1]), max([max(x.u) for x in level2]))
    plotFlow(level1, clients, 'r', mxflow)
    plotFlow(level2, level1, 'b', mxflow)


# Imprimir información sobre la ejecución de un algoritmo
def printSolution(sel_level1, sel_level2, clients, p, q, ps):
    print('Función Objetivo: {:.2f}'.format(obj_function(sel_level1, sel_level2)))
    print('Clientes: {}'.format(len(clients)))
    print('Tiempo del algoritmo: {:.3f}'.format(ps.total_tt))
    print()

# Crear una copia vacía de los objetos
def copy_solution(level1, level2, clients):
    level1_temp = [x.new_clone() for x in level1]
    level2_temp = [x.new_clone() for x in level2]
    clients_temp = [x.new_clone() for x in clients]
    return level1_temp, level2_temp, clients_temp

def save_profile_file(filename, pr):
    profile_file = open(filename+'.profile', 'w')
    ps = pstats.Stats(pr, stream=profile_file).strip_dirs().sort_stats('cumtime').print_stats()
    profile_file.close()
    return ps

def mergeSolutions(sel, empty):
    sel1_indices = {x.i:x for x in sel.level1}
    level1 = []
    for i in range(len(empty.level1)):
        if i in sel1_indices:
            loc = sel1_indices[i]
            loc.is_in = True
            level1.append(loc)
        else:
            level1.append(empty.level1[i].new_clone())

    sel2_indices = {x.i:x for x in sel.level2}
    level2 = []
    for i in range(len(empty.level2)):
        if i in sel2_indices:
            loc = sel2_indices[i]
            loc.is_in = True
            level2.append(loc)
        else:
            level2.append(empty.level2[i].new_clone())

    return Solution(level1, level2, empty.clients, empty.p, empty.q)
