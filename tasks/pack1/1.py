from random import randint

number = randint(100, 999)  # Генерация числа
print(number)  # Вывод числа
print(number // 100 % 10 + number // 10 % 10 + number % 10)  # Вывод суммы его цифр

