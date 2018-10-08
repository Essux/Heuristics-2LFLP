class Solution:
    def __init__(self, level1 = [], level2 = [], clients = [], p = 0, q = 0):
        self.level1 = level1
        self.level2 = level2
        self.clients = clients
        self.p = p
        self.q = q

    def copy_solution(self):
        level1 = [x.new_clone() for x in self.level1]
        level2 = [x.new_clone() for x in self.level2]
        clients = [x.new_clone() for x in self.clients]
        return Solution(level1, level2, clients, self.p, self.q)


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

    def __repr__(self):
        return "<Client Instance {}: x {} y {} d {} sd {}>".format(
                    self.i, self.x, self.y, self.d, self.sd)

    def new_clone(self):
        new_obj = Client(self.x, self.y, self.d, self.i)
        return new_obj

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
        self.cSum = 0
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