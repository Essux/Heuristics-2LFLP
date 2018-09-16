from math import inf
from itertools import product

# Ejecuta el método constructivo y retorna las instalaciones seleccionadas,
# los clientes y el valor de la función objetivo de la solución
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

    # Cálculo de la función objetivo
    func_obj = 0

    for l in level1:
        for cap, cost in zip(l.u, l.c):
            # Suma del flujo entre cada pareja por el costo
            # unitario de transporte
            func_obj += cap * cost

    for l in level2:
        for cap, cost in zip(l.u, l.c):
            func_obj += cap * cost

    return level1, level2, clients, func_obj

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
    assert len(adj)-1 == sink

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

    return flow, fcost