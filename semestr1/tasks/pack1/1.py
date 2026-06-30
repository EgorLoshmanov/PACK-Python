from random import randint

number = randint(100, 999)  
print(number)  
print(number // 100 % 10 + number // 10 % 10 + number % 10)  

