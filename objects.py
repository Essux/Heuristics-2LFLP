class Client:
    def __init__(self, x, y, d, i):
        # Coordinates
        self.x = x
        self.y = y
        # Demand
        self.d = d
        # Satisfied demand
        self.sd = 0
        # Identifier
        self.i = i

class Facility:
    def __init__(self, x, y, m, i, n_fac):
        # Coordinates
        self.x = x
        self.y = y
        # Capacity
        self.m = m
        # Identifier
        self.i = i
        # Included/Excluded
        self.is_in = 0
        # Costs
        self.c = [0 for x in range(n_fac)]
        # OutFlow
        self.u = [0 for x in range(n_fac)]
        self.uSum = 0
        # InFlow
        self.inflow = 0
