from random import randint
import sys

LEVEL_MAX = 20
LEVEL_MIN = 13
CLIENT_MAX = 12
CLIENT_MIN = 8
CAP_MAX = 200
COOR_MAX = 1000
COOR_MIN = 0

def generate(file):
    sys.stdout = open(file, 'w')
    I = randint(CLIENT_MIN, CLIENT_MAX)
    I1 = I
    print(I)
    clSum = 0
    for i in range(I):
        c = randint(1, CAP_MAX)
        clSum += c
        print(randint(COOR_MIN, COOR_MAX), randint(COOR_MIN, COOR_MAX), c)

    I = randint(LEVEL_MIN, LEVEL_MAX)
    p = randint(LEVEL_MIN, I)
    print(I, p)
    cs = []
    for i in range(I-1):
        c = randint(1, CAP_MAX)
        cs.append(c)
        print(randint(COOR_MIN, COOR_MAX), randint(COOR_MIN, COOR_MAX), c)
    lSum = sum(sorted(cs, reverse=True)[:p-1])
    print(randint(COOR_MIN, COOR_MAX), randint(COOR_MIN, COOR_MAX), max(randint(1, CAP_MAX), clSum-lSum))

    I = randint(LEVEL_MIN, LEVEL_MAX)
    p = randint(LEVEL_MIN, I)
    print(I, p)
    cs = []
    for i in range(I-1):
        c = randint(1, CAP_MAX)
        cs.append(c)
        print(randint(COOR_MIN, COOR_MAX), randint(COOR_MIN, COOR_MAX), c)
    lSum = sum(sorted(cs, reverse=True)[:p-1])
    print(randint(COOR_MIN, COOR_MAX), randint(COOR_MIN, COOR_MAX), max(randint(1, CAP_MAX), clSum-lSum))
    sys.stdout.close()
    sys.stdout = sys.__stdout__
