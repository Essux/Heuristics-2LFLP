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


def executeMethod(op):
    global level1, level2, clients, p, q
    if op == 0:
        print('Constructive Method')
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return constructive_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 1:
        print('Random + MaxFlow')
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return random_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 2:
        print('Greedy + MaxFlow')
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return average_cost_method(level1_temp, level2_temp, clients_temp, p, q)
    elif op == 3:
        print('Noise + Greedy + MaxFlow')
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return noise_costs(level1_temp, level2_temp, clients_temp, p, q)

# 0 -> Constructive
# 1 -> Random
# 2 -> Greedy
# 3 -> Noise + Greedy
CHOOSE = [0, 1, 2, 3]

for op in CHOOSE:
    try:
        # Medir el tiempo de ejecución del algoritmo
        pr = cProfile.Profile()
        pr.enable()

        sel_level1, sel_level2, clients = executeMethod(op)

        pr.disable()
        profile_file = open(filename+'.profile', 'w')
        ps = pstats.Stats(pr, stream=profile_file).strip_dirs().sort_stats('cumtime').print_stats()
        profile_file.close()

        # Graficar entradas
        plotAnnot(level1, 'b')
        plotAnnot(clients, 'r')
        plotAnnot(level2, 'g')

        printSolution(sel_level1, sel_level2, clients, p, q, ps)
        plotSolution(sel_level1, sel_level2, clients)
    except AssertionError:
        print('The algorithm was not able to find a feasible solution')