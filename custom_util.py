from math import sqrt, inf

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