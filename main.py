from objects import Facility, Client
from readinput import read, model_export
from math import sqrt, inf
from matplotlib import pyplot as plt
from random_generator import generate
from algorithm import *
import cProfile, pstats
from custom_util import *

filename = 'I1'
# Generar datos aleatorios
generate(filename + '.in')
# Leer archivo
clients, level1, level2, p, q = read(filename + '.in')
# Calcular pesos entre puntos
calculateWeights(level1, level2, clients)
# Exportar input para CPLEX
#model_export(filename + '.dat', clients, level1, level2, p, q)

METHOD_NAMES = ['Constructive Method', 'Random + MaxFlow', 'Greedy + MaxFlow', 'Noise + Greedy + MaxFlow', 'RCL']

def executeMethod(op):
    global level1, level2, clients, p, q
    print(METHOD_NAMES[op])
    if op == 0:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return constructive_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 1:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return random_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 2:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return average_cost_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 3:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return noise_costs(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 4:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return rcl_constructive(level1_temp, level2_temp, clients_temp, p, q, k = 20)

# 0 -> Constructive
# 1 -> Random
# 2 -> Greedy
# 3 -> Noise + Greedy
# 4 -> RCL
CHOOSE = [0, 1, 2, 3, 4]

for op in CHOOSE:
    try:
        # Medir el tiempo de ejecución del algoritmo
        pr = cProfile.Profile()
        pr.enable()

        sel_level1, sel_level2, clients = executeMethod(op)

        pr.disable()
        ps = save_profile_file(filename, pr)

        # Graficar entradas
        plotAnnot(level1, 'b')
        plotAnnot(clients, 'r')
        plotAnnot(level2, 'g')
        plt.title('{}: {:,.0f}'.format(METHOD_NAMES[op], obj_function(sel_level1, sel_level2)))

        printSolution(sel_level1, sel_level2, clients, p, q, ps)
        plotSolution(sel_level1, sel_level2, clients)

        #plt.show()
        plt.savefig('imgs/'+METHOD_NAMES[op])
        plt.clf()
    except AssertionError:
        print('The algorithm was not able to find a feasible solution')