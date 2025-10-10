from math import sqrt

def isPrime(number):
    if number == 0 or number == 1:
        return 0
    for i in range(2, int(sqrt(number)) + 1):
        if number % i == 0:
            return 0
    return 1

lst = []
N = int(input())

for i in range(1, N + 1):
    if isPrime(i) == 1:
        lst.append(i)

print(lst)