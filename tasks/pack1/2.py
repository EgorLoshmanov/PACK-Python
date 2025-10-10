from random import randint

lst = []
number = randint(1, 10 ** 10)

for i in str(number):
    lst.append(int(i))

print(number)
print(sum(lst))
