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

    def __repr__(self):
        return "<Facility Instance {}: x {} y {} m {} in {} u_len {} u_sum {} inflow {}>".format(
                    self.i, self.x, self.y, self.m, self.is_in, len(self.u), sum(self.u), self.inflow)

    def new_clone(self):
        new_obj = Facility(self.x, self.y, self.m, self.i, len(self.u))
        new_obj.c = self.c
        return new_obj