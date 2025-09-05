import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("N", type=int)
arg = parser.parse_args()

lst = []
for i in range(arg.N):
    lst.append(random.randint(-999, 999))

for _ in range(len(lst)):
    for j in range(len(lst) - 1):
        if lst[j + 1] < lst[j]:
            lst[j], lst[j + 1] = lst[j + 1], lst[j]

print(lst)