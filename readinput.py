import sys
from objects import Facility, Client

def read(file):
    sys.stdin = open(file)
    I = int(input())
    clients = []
    for i in range(I):
        x, y, d = map(int, input().split())
        clients.append(Client(x, y, d, i))

    J, p = map(int, input().split())

    level1 = []
    for j in range(J):
        x, y, m = map(int, input().split())
        f = Facility(x, y, m, j, I)
        f.cSum = sum(f.c)
        level1.append(f)

    K, q = map(int, input().split())
    level2 = []
    for k in range(K):
        x, y, m = map(int, input().split())
        f = Facility(x, y, m, k, J)
        f.cSum = sum(f.c)
        level2.append(f)

    sys.stdin.close()
    sys.stdin = sys.__stdin__
    return (clients, level1, level2, p, q)

# Exportar datos para ser usados en AMPL
def model_export(file, clients, level1, level2, p, q):
    sys.stdout = open(file, 'w')

    print('param n := {};'.format(len(clients)))
    print('param m := {};'.format(len(level1)))
    print('param o := {};'.format(len(level2)))

    print('param c1: ', end='')
    for i in range(len(clients)):
        print(i+1, end=' ')
    print(':=')
    for i, l in enumerate(level1):
        print(i+1, end=' ')
        for c in l.c:
            print(c, end=' ')
        print()
    print(';')

    print('param c2: ', end='')
    for i in range(len(level1)):
        print(i+1, end=' ')
    print(':=')
    for i, l in enumerate(level2):
        print(i+1, end=' ')
        for c in l.c:
            print(c, end=' ')
        print()
    print(';')

    print('param p := {};'.format(p))
    print('param q := {};'.format(q))

    print('param d :=')
    for i, c in enumerate(clients):
        print(i+1, c.d)
    print(';')

    print('param M1 :=')
    for i, c in enumerate(level1):
        print(i+1, c.m)
    print(';')

    print('param M2 :=')
    for i, c in enumerate(level2):
        print(i+1, c.m)
    print(';')

    sys.stdout.close()
    sys.stdout = sys.__stdout__