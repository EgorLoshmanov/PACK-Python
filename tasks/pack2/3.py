import random

N = int(input())

lst = []
for i in range(N):
    lst.append(random.randint(-999, 999))

even = [x for x in lst if x % 2 == 0]
odd = [x for x in lst if x % 2 != 0]

print(lst)
print(even)
print(odd)
print('-' * 50)
print(f"Чётных чисел: {len(even)}")
print(f"Нечётных чисел: {len(odd)}")