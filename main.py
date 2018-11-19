from objects import Facility, Client, Solution
from readinput import read, model_export
from random_generator import generate
from algorithm import *
from custom_util import *

from math import sqrt, inf
from matplotlib import pyplot as plt
import cProfile, pstats
import sys

if len(sys.argv) == 1:
    LEVEL_MAX = 50
    LEVEL_MIN = 30
    CLIENT_MAX = 20
    CLIENT_MIN = 20
else:
    LEVEL_MAX = int(sys.argv[1])
    LEVEL_MIN = int(sys.argv[2])
    CLIENT_MAX = int(sys.argv[3])
    CLIENT_MIN = int(sys.argv[4])

filename = 'I1'
# Generar datos aleatorios
generate(filename + '.in', LEVEL_MAX, LEVEL_MIN, CLIENT_MAX, CLIENT_MIN)
# Leer archivo
clients, level1, level2, p, q = read(filename + '.in')
# Calcular pesos entre puntos
calculateWeights(level1, level2, clients)
# Exportar input para CPLEX
model_export(filename + '.dat', clients, level1, level2, p, q)

METHOD_NAMES = ['Constructive Method', 'Random + MaxFlow', 'Greedy + MaxFlow', 'Noise + Greedy + MaxFlow', 'RCL', '', 'RCL2']
SEARCH_NAMES = ['', 'Local Search', 'VND', 'GRASP', 'Genetic Algorithm']

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
    elif op == 5:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return level1_temp, level2_temp, clients_temp
    elif op == 6:
        level1_temp, level2_temp, clients_temp = copy_solution(level1, level2, clients)
        return rcl_constructive2(level1_temp, level2_temp, clients_temp, p, q, k = 5)

def execute_search_method(op, sel_level1, sel_level2, clients, p, q):
    sel_solution = Solution(sel_level1, sel_level2, clients, p, q)
    empty_solution = Solution(level1, level2, clients, p, q)
    full_solution = mergeSolutions(sel_solution, empty_solution)

    print(SEARCH_NAMES[op])
    if op == 0:
        return full_solution
    elif op == 1:
        return local_search(full_solution)
    elif op == 2:
        return variable_neighborhood_descent(full_solution)
    elif op == 3:
        return grasp(full_solution)
    elif op == 4:
        return genetic_algorithm(full_solution, 100, 0.5)

# 0 -> Constructive
# 1 -> Random
# 2 -> Greedy
# 3 -> Noise + Greedy
# 4 -> RCL
# 5 -> Do nothing
# 6 -> RCL2
CHOOSE_INITIAL = [3, 5]

# 0 -> Do nothing
# 1 -> Local search
# 2 -> VND
# 3 -> GRASP
# 4 -> Genetic Algorithm
CHOOSE_SEARCH = [2, 4]

for op, op_search in zip(CHOOSE_INITIAL, CHOOSE_SEARCH):
    try:
        # Medir el tiempo de ejecución del algoritmo
        pr = cProfile.Profile()
        pr.enable()

        sel_level1, sel_level2, clients = executeMethod(op)

        sol = execute_search_method(op_search, sel_level1, sel_level2, clients, p, q)

        sel_level1, sel_level2, clients = sol.level1, sol.level2, clients

        pr.disable()
        ps = save_profile_file(filename, pr)

        # Graficar entradas
        plotAnnot(level1, 'b')
        plotAnnot(clients, 'r')
        plotAnnot(level2, 'g')
        plt.title('{}: {:,.0f}'.format(METHOD_NAMES[op]+'+'+SEARCH_NAMES[op_search], obj_function(sel_level1, sel_level2)))

        printSolution(sel_level1, sel_level2, clients, p, q, ps)
        plotSolution(sel_level1, sel_level2, clients)

        plt.show()
        plt.savefig('imgs/'+METHOD_NAMES[op])
        plt.clf()
    except AssertionError:
        print('The algorithm was not able to find a feasible solution')
