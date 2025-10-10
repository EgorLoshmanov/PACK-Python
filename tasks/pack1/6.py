X, Y = int(input()), int(input())

for i in range(1, Y + 1):
    X += X * 0.1

print(X)