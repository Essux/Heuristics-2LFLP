class Solution:
    def __init__(self, level1 = [], level2 = [], clients = [], p = 0, q = 0):
        self.level1 = level1
        self.level2 = level2
        self.clients = clients
        self.p = p
        self.q = q

    def copy_solution(self, copyFlow=False):
        level1 = [x.new_clone(copyFlow=copyFlow) for x in self.level1]
        level2 = [x.new_clone(copyFlow=copyFlow) for x in self.level2]
        clients = [x.new_clone() for x in self.clients]
        return Solution(level1, level2, clients, self.p, self.q)

    def __repr__(self):
        l1_str = '\n'.join([x.__repr__() for x in self.level1])
        l2_str = '\n'.join([x.__repr__() for x in self.level2])
        clients_str = '\n'.join([x.__repr__() for x in self.clients])
        return "<Solution Instance: p {} q {}\nLevel 1:\n{}\nLevel 2\n{}\nClients:\n{}>".format(self.p, self.q,
                l1_str, l2_str, clients_str)


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
        self.is_in = False
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

    def new_clone(self, copyFlow=False):
        new_obj = Facility(self.x, self.y, self.m, self.i, len(self.u))
        new_obj.c = self.c
        new_obj.is_in = self.is_in
        if copyFlow:
            import copy
            new_obj.u = copy.deepcopy(self.u)
        return new_obj