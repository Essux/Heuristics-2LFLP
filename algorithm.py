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

