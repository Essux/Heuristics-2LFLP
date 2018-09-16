from objects import Facility, Client
from readinput import read, model_export
from math import sqrt, inf
from matplotlib import pyplot as plt
from random_generator import generate
from algorithm import constructive_method
import cProfile, pstats


## Funciones Auxiliares ##

# Calcular pesos entre todas las parejas de puntos
def calculateWeights():
    for l in level1:
        for i in range(len(l.c)):
            l.c[i] = dist(l, clients[i])

    for l in level2:
        for i in range(len(l.c)):
            l.c[i] = dist(l, level1[i])

# Retorna la distancia entre dos puntos
def dist(a, b):
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

# Graficar un conjunto de instalaciones
def plotAnnot(li, color):
    plt.scatter([x.x for x in li], [x.y for x in li], color=color)
    for l in li:
        plt.annotate(l.i+1, (l.x+0.5, l.y+0.5))

# Graficar flujo entre instalaciones
def plotFlow(li1, li2, color):
    for l in li1:
        for c in li2:
            f = l.u[c.i]
            if f > 0: plt.plot([l.x, c.x], [l.y, c.y], color = color, linewidth=f/mxflow*3)


## Inicio del script ##

filename = 'I1'
# Generar datos aleatorios
generate(filename + '.in')
# Leer archivo
clients, level1, level2, p, q = read(filename + '.in')
# Calcular pesos entre puntos
calculateWeights()
# Exportar input para CPLEX
model_export(filename + '.dat', clients, level1, level2, p, q)

# Graficar entradas
plotAnnot(level1, 'b')
plotAnnot(clients, 'r')
plotAnnot(level2, 'g')

# Medir el tiempo de ejecución del algoritmo
pr = cProfile.Profile()
pr.enable()
# Ejecutar el algoritmo
sel_level1, sel_level2, clients, func_obj = constructive_method(level1, level2, clients, p, q)
pr.disable()
profile_file = open(filename+'.profile', 'w')
ps = pstats.Stats(pr, stream=profile_file).strip_dirs().sort_stats('cumtime').print_stats()
profile_file.close()


print('Función Objetivo: {:.2f}'.format(func_obj))
print('Clientes: {}'.format(len(clients)))
print('Instalaciones Usadas de Nivel 1: {}/{}'.format(p, len(level1)))
print('Instalaciones Usadas de Nivel 2: {}/{}'.format(q, len(level2)))
print('Tiempo del algoritmo: {:.3f}'.format(ps.total_tt))

# Graficar solución
mxflow = max(max([max(x.u) for x in level1]), max([max(x.u) for x in level2]))
plotFlow(level1, clients, 'r')
plotFlow(level2, level1, 'b')

plt.show()