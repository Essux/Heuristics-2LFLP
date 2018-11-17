from subprocess import call


#Level Max, Level Min, Client Max, Client Min
PARAMS = [
    [50, 30, 20, 20],
    [120, 80, 50, 50],
    [240, 160, 100, 100],
    [500, 300, 200, 200]
]

i = 0
while True:
    par = list(map(str, PARAMS[i]))
    print(par)
    call(["python", "main.py"] + par)

    i = (i+1)%len(PARAMS)