from math import inf, sqrt
from itertools import product, combinations, combinations_with_replacement, accumulate
from random import sample, normalvariate, randrange, random
from custom_util import obj_function, copy_solution, mergeSolutions
from objects import Solution, RandomKeySolution
from collections import deque
from bisect import bisect
import time

"""
 Selecciona las instalaciones de mayor capacidad y
 asigna los flujos de material con menor coste
"""
def constructive_method(level1, level2, clients, p, q):
    # Tomar las p instalaciones de nivel 1 con más capacidad
    level1 = sorted(level1, key=lambda x : x.m, reverse=True)[:p]
    for l in level1:
        l.is_in = 1

    # Tomar las q instalaciones de nivel 2 con más capacidad
    level2 = sorted(level2, key=lambda x : x.m, reverse=True)[:q]
    for l in level2:
        l.is_in = 1

    # Generar todas las parejas de instalaciones de nivel 1 con clientes
    con1 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level1, clients)]

    # Ordenar las parejas de menor a mayor costo
    con1.sort(key=lambda x : x[0])

    for t in con1:
        l = t[1]
        cl = t[2]
        # Llevar el máximo material posible entre el cliente y la instalación
        flow = min(cl.d-cl.sd, l.m-l.uSum)
        # Actualizar la demanda satisfecha del cliente
        cl.sd += flow
        # Actualizar el flujo de material saliente de la instalación
        l.u[cl.i] += flow
        l.uSum += flow

    # Repetir un proceso similar al anterior pero tomando como clientes
    # a las instalaciones de nivel 1
    con2 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level2, level1)]

    # Ordenar las parejas de menor a mayor costo
    con2.sort(key=lambda x : x[0])

    for t in con2:
        l = t[1]
        cl = t[2]
        # Llevar el máximo material posible entre instalaciones
        flow = min(cl.uSum-cl.inflow, l.m-l.uSum)
        # Actualizar el flujo entrante de la instalación de nivel 1
        cl.inflow += flow
        # Actualizar el flujo de material saliente de la instalación de nivel 2
        l.u[cl.i] += flow
        l.uSum += flow

    for cl in clients:
        assert(cl.d==cl.sd)

    return level1, level2, clients

"""
 Retorna la asignación óptima de un conjunto de instalaciones
 usando el algoritmo de Edmonds Karp con relabelling y el algoritmo
 de Dijkstra
"""
def min_cost_max_flow(level1, level2, clients, p, q):
    obj_to_i = {}
    i_to_obj = {}
    index = 1
    # Fuente del grafo
    source = 0
    # Asignar a instalaciones de nivel 2 los indices del [1-q]
    for l2 in level2:
        obj_to_i[l2] = index
        i_to_obj[index] = l2
        index += 1

    # Asignar a instalaciones de nivel 1 los indices del [q+1, q+p]
    for l1 in level1:
        obj_to_i[l1] = index
        i_to_obj[index] = l1
        index += 1

    # Asignar segundo nodo de instalaciones de nivel 1 los indices del [q+p+1, q+2p]
    for l1 in level1:
        i_to_obj[index] = l1
        index += 1

    # Asignar a instalaciones de nivel 1 los indices del [q+2p+1, q+2p+i]
    # dónde i es el numero de clientes
    for cl in clients:
        obj_to_i[cl] = index
        i_to_obj[index] = cl
        index += 1

    # Sumidero del grafo
    sink = index
    index += 1

    # Numero total de nodos
    n = index
    # Matriz de capacidades
    cap = [[0 for x in range(n)] for x in range(n)]
    # Matriz de costos
    cost = [[0 for x in range(n)] for x in range(n)]

    # Añadir aristas entre la fuente y las instalaciones de nivel 2
    # Cap: Capacidad de las instalaciones
    # Costo: 0
    for l2 in level2:
        cap[source][obj_to_i[l2]] = l2.m

    # Añadir aristas entre el nivel 2 y el 1
    # Cap: Infinito
    # Costo: Costo entre instalaciones
    for l2 in level2:
        for l1 in level1:
            i = obj_to_i[l2]
            j = obj_to_i[l1]
            cap[i][j] = inf
            cost[i][j] = l2.c[l1.i]

    # Añadir aristas de cada instalación de nivel 2 al segundo nodo de la instalación
    # Cap: Capacidad de las instalaciones
    # Costo: 0
    for l1 in level1:
        i = obj_to_i[l1]
        cap[i][i+p] = l1.m

    # Añadir aristas entre del segundo nodo del nivel 1 y los clientes
    # Cap: Infinito
    # Costo: Costo unitario de envío entre instalaciones
    for l1 in level1:
        for cl in clients:
            i = obj_to_i[l1]+p
            j = obj_to_i[cl]
            cap[i][j] = inf
            cost[i][j] = l1.c[cl.i]


    # Añadir aristas entre los clientes y el sumidero
    # Cap: Demanda de los clientes
    # Costo: 0
    for cl in clients:
        cap[obj_to_i[cl]][sink] = cl.d

    # Lista de adyacencia
    adj = [[] for x in range(n)]
    # Grafo residual
    fnet = [[0 for x in range(n)] for x in range(n)]
    # Función de labelling
    pi = [0 for x in range(n)]
    # Flujo máximo
    flow = 0
    # Mínimo costo
    fcost = 0
    # Padre de cada nodo
    par = []

    ### Inicio del algoritmo MinCostMaxFlow ###
    for i in range(n):
        for j in range(n):
            if cap[i][j] or cap[j][i]:
                adj[i].append(j)

    def dijkstra():
        nonlocal par
        d = [inf for x in range(n)]
        par = [-1 for x in range(n)]

        d[source] = 0
        par[source] = -n - 1

        # Función de potencial
        def pot(u, v):
            return d[u] + pi[u] - pi[v]

        while True:
            # Encontrar la menor distancia
            u = -1
            bestD = inf
            for i in range(n):
                if par[i] < 0 and d[i] < bestD:
                    bestD = d[i]
                    u = i

            if bestD == inf: break

            # Relajar todas las aristas (u, i) y (i, u)
            par[u] = -par[u] - 1
            for v in adj[u]:
                # Intentar deshacer arista v->u
                if par[v] >= 0: continue
                if fnet[v][u] and d[v] > pot(u, v) - cost[v][u]:
                    d[v] = pot(u,v) - cost[v][u]
                    par[v] = -u-1

                # Intentar arista u->v
                if (fnet[u][v] < cap[u][v]) and (d[v] > pot(u, v) + cost[u][v]):
                    d[v] = pot(u, v) + cost[u][v]
                    par[v] = -u-1

        for i in range(n):
            if pi[i] < inf: pi[i] += d[i]

        return par[sink] >= 0


    while dijkstra():
        # Encontrar el cuello de botella de la capacidad
        bot = inf
        v = sink
        u = par[v]
        while v!=source:
            c1 = fnet[v][u] if fnet[v][u] else cap[u][v] - fnet[u][v]
            bot = min(c1, bot)
            v = u
            u = par[v]

        # Actualizar grafo residual
        v = sink
        u = par[v]
        while v!=source:
            if fnet[v][u]:
                fnet[v][u] -= bot
                fcost -= bot * cost[v][u]
            else:
                fnet[u][v] += bot
                fcost += bot * cost[u][v]
            v = u
            u = par[v]

        flow += bot

    for i in range(1, q+1):
        for j in range(q+1, q+p+1):
            level2[i-1].u[i_to_obj[j].i] = fnet[i][j] - fnet[j][i]

    for i in range(q+p+1, q+2*p+1):
        for j in range(q+2*p+1, q+2*p+1+len(clients)):
            level1[i-(q+p+1)].u[i_to_obj[j].i]= fnet[i][j] - fnet[j][i]

    for i in range(q+2*p+1, q+2*p+1+len(clients)):
        clients[i_to_obj[i].i].sd = fnet[i][sink] - fnet[sink][i]

    return level1, level2, clients, flow, fcost


"""
 Selecciona aleatoriamente p y q instalaciones y
 retorna su asignación óptima usando MinCostMaxFlow
"""
def random_method(level1, level2, clients, p, q, iterations=1):
    bestObj = inf
    ans_sel_level1, ans_sel_level2, ans_clients = None, None, None

    for i in range(iterations):
        # Elegir aleatoriamente p instalaciones
        sel_level1 = sample(level1, p)
        sel_level2 = sample(level2, q)
        sel_level1 = list(map(lambda x : x.new_clone(), sel_level1))
        sel_level2 = list(map(lambda x : x.new_clone(), sel_level2))

        # Calcular la asignación óptima usando MinCostMaxFlow
        cand_sel_level1, cand_sel_level2, cand_clients, flow, fcost = min_cost_max_flow(sel_level1, sel_level2, clients, p, q)
        cand_obj = obj_function(cand_sel_level1, cand_sel_level2)

        # Omit unfeasible solutions
        if flow != sum([x.d for x in clients]):
            continue

        if cand_obj < bestObj:
            ans_sel_level1, ans_sel_level2, ans_clients = cand_sel_level1, cand_sel_level2, cand_clients
            bestObj = cand_obj

    assert(ans_sel_level1 is not None)

    return ans_sel_level1, ans_sel_level2, ans_clients

"""
 Selecciona las instalaciones con menor costo promedio
 y retorna su asignación óptima usando MinCostMaxFlow
"""
def average_cost_method(level1, level2, clients, p, q):
    # Garantizar que la suma de los costos esté actualizada
    if level1[0].cSum == 0:
        for l1 in level1:
            l1.cSum = sum(l1.c)
    # Seleccionar las primeras instalaciones con menor costo promedio
    level1.sort(key=lambda x : x.cSum)
    sel_level1 = level1[:p]

    # Garantizar que la suma de los costos esté actualizada
    if level2[0].cSum == 0:
        for l2 in level2:
            l2.cSum = sum(l2.c)
    level2.sort(key=lambda x : x.cSum)
    sel_level2 = level2[:q]

    # Calcular la asignación óptima usando MinCostMaxFlow
    sel_level1, sel_level2, clients, flow, fcost = min_cost_max_flow(sel_level1, sel_level2, clients, p, q)
    assert(flow == sum([x.d for x in clients]))

    return sel_level1, sel_level2, clients

"""
 Selecciona las instalaciones con menor costo promedio
 añadiendo un ruido aleatorio normal al costo y
 retorna su asignación óptima usando MinCostMaxFlow
"""
def noise_costs(level1, level2, clients, p, q):
    l1_costs = [sum(x.c)/len(x.c) for x in level1]
    mean = sum(l1_costs) / len(l1_costs)
    sd = sqrt(sum([(x-mean)**2 for x in l1_costs]) / (len(l1_costs) - 1))
    for x in level1:
        x.cSum = x.cSum + normalvariate(0, sd)

    l2_costs = [sum(x.c)/len(x.c) for x in level2]
    mean = sum(l2_costs) / len(l2_costs)
    sd = sqrt(sum([(x-mean)**2 for x in l2_costs]) / (len(l2_costs) - 1))
    for x in level2:
        x.cSum = x.cSum + normalvariate(0, sd)

    return average_cost_method(level1, level2, clients, p, q)


"""
 Selecciona instalaciones aleatoriamente y
 asigna los flujos de material con menor coste
"""
def rcl_constructive(level1, level2, clients, p, q, k=5):
    # Tomar p instalaciones de nivel 1 aleatoriamente
    level1 = sample(level1, p)
    # Tomar q instalaciones de nivel 2 aleatoriamente
    level2 = sample(level2, q)

    # Generar todas las parejas de instalaciones de nivel 1 con clientes
    con1 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level1, clients)]

    # Ordenar las parejas de menor a mayor costo
    con1.sort(key=lambda x : x[0])
    con1 = deque(con1)

    rcl1 = []

    while con1 or rcl1:
        # Añadir las aristas faltantes a la rcl para que alcance tamaño k
        while len(rcl1) < k and con1:
            rcl1.append(con1.popleft())

        # Seleccionar una arista aleatoriamente de la rcl
        sel_i = randrange(len(rcl1))
        sel = rcl1[sel_i]
        del rcl1[sel_i]

        l = sel[1]
        cl = sel[2]
        # Llevar el máximo material posible entre el cliente y la instalación
        flow = min(cl.d-cl.sd, l.m-l.uSum)
        # Actualizar la demanda satisfecha del cliente
        cl.sd += flow
        # Actualizar el flujo de material saliente de la instalación
        l.u[cl.i] += flow
        l.uSum += flow

    # Repetir un proceso similar al anterior pero tomando como clientes
    # a las instalaciones de nivel 1
    con2 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level2, level1)]

    con2.sort(key=lambda x : x[0])
    con2 = deque(con2)

    rcl2 = []

    while con2 or rcl2:
        while len(rcl2) < k and con2:
            rcl2.append(con2.popleft())

        sel_i = randrange(len(rcl2))
        sel = rcl2[sel_i]
        del rcl2[sel_i]

        l = sel[1]
        cl = sel[2]
        # Llevar el máximo material posible entre instalaciones
        flow = min(cl.uSum-cl.inflow, l.m-l.uSum)
        # Actualizar el flujo entrante de la instalación de nivel 1
        cl.inflow += flow
        # Actualizar el flujo de material saliente de la instalación de nivel 2
        l.u[cl.i] += flow
        l.uSum += flow

    for cl in clients:
        assert(cl.d==cl.sd)

    return level1, level2, clients

"""
    Seleccionar las instalaciones con una RCL según el menor promedio de costes.
    Posteriormente asignar los flujos con una RCL de las aristas según el menor coste.
"""
def rcl_constructive2(level1, level2, clients, p, q, k):
    for l1 in level1:
        l1.cSum = sum(l1.c)
    level1.sort(key=lambda x: x.cSum)
    level1 = deque(level1)
    sel_level1 = []
    rcl1 = []
    while len(sel_level1)<p:
        while len(rcl1) < k and level1:
            rcl1.append(level1.popleft())

        sel_i = randrange(len(rcl1))
        sel = rcl1[sel_i]
        del rcl1[sel_i]

        sel_level1.append(sel)

    level1 = sel_level1

    for l2 in level2:
        l2.cSum = sum(l2.c)
    level2.sort(key=lambda x: x.cSum)
    level2 = deque(level2)
    sel_level2 = []
    rcl2 = []
    while len(sel_level2)<q:
        while len(rcl2) < k and level2:
            rcl2.append(level2.popleft())

        sel_i = randrange(len(rcl2))
        sel = rcl2[sel_i]
        del rcl2[sel_i]

        sel_level2.append(sel)

    level2 = sel_level2

    # Generar todas las parejas de instalaciones de nivel 1 con clientes
    con1 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level1, clients)]

    # Ordenar las parejas de menor a mayor costo
    con1.sort(key=lambda x : x[0])

    for t in con1:
        l = t[1]
        cl = t[2]
        # Llevar el máximo material posible entre el cliente y la instalación
        flow = min(cl.d-cl.sd, l.m-l.uSum)
        # Actualizar la demanda satisfecha del cliente
        cl.sd += flow
        # Actualizar el flujo de material saliente de la instalación
        l.u[cl.i] += flow
        l.uSum += flow

    # Repetir un proceso similar al anterior pero tomando como clientes
    # a las instalaciones de nivel 1
    con2 = [(t[0].c[t[1].i], t[0], t[1]) for t in product(level2, level1)]

    # Ordenar las parejas de menor a mayor costo
    con2.sort(key=lambda x : x[0])

    for t in con2:
        l = t[1]
        cl = t[2]
        # Llevar el máximo material posible entre instalaciones
        flow = min(cl.uSum-cl.inflow, l.m-l.uSum)
        # Actualizar el flujo entrante de la instalación de nivel 1
        cl.inflow += flow
        # Actualizar el flujo de material saliente de la instalación de nivel 2
        l.u[cl.i] += flow
        l.uSum += flow

    for cl in clients:
        assert(cl.d==cl.sd)

    return level1, level2, clients

# Asegurarse que los atributos de la solución tengan los valores que deberían
def setup_solution(sol):
    for location in sol.level1:
        location.uSum = sum(location.u)
    for location in sol.level2:
        location.uSum = sum(location.u)

"""
 Genera el vecindario de una solución con movimientos de
 quitar y poner instalaciones
"""
def facility_inout_neighborhood(sol):
    setup_solution(sol)
    initial_cost = obj_function(sol.level1, sol.level2)

    l1_in = list(filter(lambda x : x.is_in, sol.level1))
    l1_out = list(filter(lambda x : not x.is_in, sol.level1))

    # Intercambiar instalaciones de nivel 1
    for l_in in l1_in:
        for l_out in l1_out:
            if l_in.uSum <= l_out.m:
                func_obj_delta = 0
                for i in range(len(l_in.u)):
                    func_obj_delta += (l_out.u[i] - l_in.u[i]) * l_in.c[i]
                for i in range(len(l_in.u)):
                    func_obj_delta += (l_in.u[i] - l_out.u[i]) * l_out.c[i]
                for l2 in sol.level2:
                    func_obj_delta += (l2.u[l_out.i] - l2.u[l_in.i]) * l2.c[l_in.i]
                    func_obj_delta += (l2.u[l_in.i] - l2.u[l_out.i]) * l2.c[l_out.i]

                if func_obj_delta < 0:
                    next_sol = sol.copy_solution(copyFlow = True)
                    next_l_in = next_sol.level1[l_in.i]
                    next_l_out = next_sol.level1[l_out.i]
                    next_l_in.is_in = False
                    next_l_out.is_in = True
                    next_l_in.u, next_l_out.u = next_l_out.u, next_l_in.u

                    for l2 in next_sol.level2:
                        l2.u[l_in.i], l2.u[l_out.i] = l2.u[l_out.i], l2.u[l_in.i]

                else: next_sol = None

                yield next_sol, func_obj_delta


    # Intercambiar instalaciones de nivel 2
    l2_in = list(filter(lambda x : x.is_in, sol.level2))
    l2_out = list(filter(lambda x : not x.is_in, sol.level2))
    for l_in in l2_in:
        for l_out in l2_out:
            if l_in.uSum <= l_out.m:
                func_obj_delta = 0
                for i in range(len(l_in.u)):
                    func_obj_delta += (l_out.u[i] - l_in.u[i]) * l_in.c[i]
                for i in range(len(l_in.u)):
                    func_obj_delta += (l_in.u[i] - l_out.u[i]) * l_out.c[i]

                if func_obj_delta < 0:
                    next_sol = sol.copy_solution(copyFlow = True)
                    next_l_in = next_sol.level2[l_in.i]
                    next_l_out = next_sol.level2[l_out.i]
                    next_l_in.is_in = False
                    next_l_out.is_in = True
                    next_l_in.u, next_l_out.u = next_l_out.u, next_l_in.u
                else: next_sol = None

                yield next_sol, func_obj_delta


"""
 Genera el vecindario de una solución basado en movimientos de flujos
"""
def inlevel_neighborhood(sol):
    setup_solution(sol)
    initial_cost = obj_function(sol.level1, sol.level2)

    level2 = list(filter(lambda x: x.is_in, sol.level2))

    for l2 in level2:
        for l1 in sol.level1:
            if l2.u[l1.i] == 0: continue
            cur_flow = l2.u[l1.i]

            for new_l2 in level2:
                if new_l2.i == l2.i: continue
                avail_cap = new_l2.m - new_l2.uSum
                flow_delta = min(avail_cap, cur_flow)

                if flow_delta == 0: continue

                func_obj_delta = flow_delta * sol.level2[new_l2.i].c[l1.i] - flow_delta * sol.level2[l2.i].c[l1.i]
                if func_obj_delta < 0:
                    next_sol = sol.copy_solution(copyFlow=True)
                    next_sol.level2[l2.i].u[l1.i] -= flow_delta
                    next_sol.level2[new_l2.i].u[l1.i] += flow_delta
                else: next_sol = None

                yield next_sol, func_obj_delta


def local_search(sol):
    cur_cost = obj_function(sol.level1, sol.level2)
    print('initial cost: {:.2f}'.format(cur_cost))
    found_better = True
    while found_better:
        found_better = False
        for next_sol, delta in facility_inout_neighborhood(sol):
            #cand_cost = obj_function(next_sol.level1, next_sol.level2)
            cand_cost = cur_cost + delta
            if cand_cost < cur_cost:
                print('new best')
                sol = next_sol
                cur_cost = cand_cost
                found_better = True
                break

    return sol


def variable_neighborhood_descent(sol):
    print('VND')
    cur_cost = obj_function(sol.level1, sol.level2)
    found_better = True

    neighborhoods = [inlevel_neighborhood, facility_inout_neighborhood]

    j = 0
    while j < len(neighborhoods):
        best_cand = None
        best_cand_cost = inf
        for cand_sol, cost_delta in neighborhoods[j](sol):
            cand_cost = cur_cost + cost_delta
            if cand_cost < cur_cost:
                best_cand = cand_sol
                best_cand_cost = cand_cost
                break

        if best_cand_cost < cur_cost:
            j = 0
            sol = best_cand
            cur_cost = best_cand_cost
        else:
            j += 1

    return sol

def grasp(empty_sol, iterations = 10, k=5, rcl_method=rcl_constructive2):
    print('Method: {}'.format(rcl_method.__name__))
    best_sol = None
    best_obj = inf
    for i in range(iterations):
        level1_temp, level2_temp, clients_temp = copy_solution(empty_sol.level1, empty_sol.level2, empty_sol.clients)
        sel_level1, sel_level2, clients = rcl_method(level1_temp, level2_temp, clients_temp, empty_sol.p, empty_sol.q, k = k)

        sel_solution = Solution(sel_level1, sel_level2, clients, empty_sol.p, empty_sol.q)
        empty_solution = Solution(empty_sol.level1, empty_sol.level2, empty_sol.clients, empty_sol.p, empty_sol.q)
        full_solution = mergeSolutions(sel_solution, empty_solution)
        func_obj = obj_function(full_solution.level1, full_solution.level2)
        #print('GRASP it {} starts with {:,.2f}'.format(i+1, func_obj))

        sol = variable_neighborhood_descent(full_solution)
        func_obj = obj_function(sol.level1, sol.level2)
        #print('GRASP it {} ends with {:,.2f}'.format(i+1, func_obj))

        if func_obj < best_obj:
            best_sol = sol
            best_obj = func_obj

    return best_sol


#--------------- Genetic Algorithm ---------------#

def genetic_algorithm(sol, population_size, mutation_prob):
    l1_size = len(sol.level1)
    l2_size = len(sol.level2)
    clients_size = len(sol.clients)

    population = [generate_individual(l1_size, l2_size, clients_size) for x in range(population_size)]
    for ind in population:
        ind_sol = construct_from_RK(ind, sol)
        ind.func_obj = obj_function(ind_sol.level1, ind_sol.level2)
    population.sort(key=lambda x : x.func_obj)

    start_time = time.time()
    while time.time() - start_time < 60 * 0.05:
        sel_prob = normalize_obj_func(population)
        new_generation = []
        for i in range(population_size):
            par1, par2 = parent_selection(sel_prob)
            par1 = population[par1]
            par2 = population[par2]
            son = crossover(par1, par2)
            if random() < mutation_prob:
                son = mutate(son)
            new_generation.append(son)

        # Calcular la función objetivo de la nueva generación
        for ind in new_generation:
            ind_sol = construct_from_RK(ind, sol)
            ind.func_obj = obj_function(ind_sol.level1, ind_sol.level2)

        population = population + new_generation
        population.sort(key=lambda x : x.func_obj)
        population = population[:population_size]


    return variable_neighborhood_descent(construct_from_RK(population[0], sol))


# Crear un individuo inicializándolo con un RK aleatorio
def generate_individual(level1_size, level2_size, clients_size):
    rk_obj = RandomKeySolution()
    rk_obj.level1 = [random() for x in range(level1_size)]
    rk_obj.level2 = [random() for x in range(level2_size)]
    rk_obj.flows_l1 = [random() for x in range(level1_size * clients_size)]
    rk_obj.flows_l2 = [random() for x in range(level2_size * level1_size)]
    return rk_obj


def parent_selection(accum):
    s = set()
    while len(s) < 2:
        s.add(bisect(accum, random()))
    return tuple(s)


def normalize_obj_func(population):
    fitness = [1/x.func_obj for x in population]
    sum_fitness = sum(fitness)
    fitness = [x/sum_fitness for x in fitness]
    fitness = list(accumulate(fitness))
    return fitness


def crossover(p1, p2):
    son = RandomKeySolution()

    def uniform_fill(arr1, arr2):
        arr_son = []
        for x, y in zip(arr1, arr2):
            val = x if random() < 0.5 else y
            arr_son.append(val)
        return arr_son

    son.level1 = uniform_fill(p1.level1, p2.level1)
    son.level2 = uniform_fill(p1.level2, p2.level2)
    son.flows_l1 = uniform_fill(p1.flows_l1, p2.flows_l1)
    son.flows_l2 = uniform_fill(p1.flows_l2, p2.flows_l2)
    return son


def mutate(ind, gene_mutations=5):
    sel_id = randrange(4)
    sel_arr = [ind.level1, ind.level2, ind.flows_l1, ind.flows_l2][sel_id]
    if sel_id>1: gene_mutations *= 20
    for i in range(gene_mutations):
        sel_arr[randrange(len(sel_arr))] = random()

    return ind


# Reconstruir la solución a partir del Random Key
def construct_from_RK(rk_sol, sol):
    level1, level2, clients = copy_solution(sol.level1, sol.level2, sol.clients)
    l1_size = len(level1)
    l2_size = len(level2)
    clients_size = len(clients)

    # Garantizar que ninguna instalación esté seleccionada
    for l1 in level1:
        l1.is_in = False

    # Seleccionar las primeras p que tengan menor random key
    sel_level1 = [(rk_sol.level1[i], i) for i in range(l1_size)]
    sel_level1.sort()
    for i in range(sol.p):
        idx = sel_level1[i][1]
        level1[idx].is_in = True

    # Garantizar que ninguna instalación esté seleccionada
    for l2 in level2:
        l2.is_in = False

    # Seleccionar las primeras q que tengan menor random key
    sel_level2 = [(rk_sol.level2[i], i) for i in range(l2_size)]
    sel_level2.sort()
    for i in range(sol.q):
        idx = sel_level2[i][1]
        level2[idx].is_in = True

    # Ordenar las aristas de menor a mayor y hallar qué vértices conectan
    sel_flows_l1 = [(rk_sol.flows_l1[i], i//clients_size, i%clients_size) for i in range(len(rk_sol.flows_l1))]
    sel_flows_l1.sort()

    for t in sel_flows_l1:
        l = level1[t[1]]
        cl = clients[t[2]]
        if not l.is_in: continue
        # Llevar el máximo material posible entre el cliente y la instalación
        flow = min(cl.d-cl.sd, l.m-l.uSum)
        # Actualizar la demanda satisfecha del cliente
        cl.sd += flow
        # Actualizar el flujo de material saliente de la instalación
        l.u[cl.i] += flow
        l.uSum += flow

    # Ordenar las aristas de menor a mayor y hallar qué vértices conectan
    sel_flows_l2 = [(rk_sol.flows_l2[i], i//l1_size, i%l1_size) for i in range(len(rk_sol.flows_l2))]
    sel_flows_l2.sort()

    for t in sel_flows_l2:
        l = level2[t[1]]
        cl = level1[t[2]]
        if not l.is_in: continue
        # Llevar el máximo material posible entre instalaciones
        flow = min(cl.uSum-cl.inflow, l.m-l.uSum)
        # Actualizar el flujo entrante de la instalación de nivel 1
        cl.inflow += flow
        # Actualizar el flujo de material saliente de la instalación de nivel 2
        l.u[cl.i] += flow
        l.uSum += flow

    return Solution(level1=level1, level2=level2, clients=clients, p=sol.p, q=sol.q)