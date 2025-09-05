import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("h", type=int)
arg = parser.parse_args()

P = []
for i in range(arg.h):
    row = [1] * (i + 1)
    for j in range(i + 1):
        if j != 0 and j != i:
            row[j] = P[i-1][j-1] + P[i-1][j]

    P.append(row)

for i, r in enumerate(P):
    # количество пробелов = высота треугольника - номер строки - 1
    print(' ' * (arg.h - i - 1), end='')
    print(' '.join(map(str, r)))