matrix1, matrix2 = [], []

f = open("matrix.txt")
for line in f:
    if line[0] != '\n':
     # Разделяем строку на элементы и преобразуем в числа
        row = [int(x) for x in line.strip().split()]
        matrix1.append(row)
    else:
        break

for line in f:
    row = [int(x) for x in line.strip().split()]
    matrix2.append(row)

m, k1, k2 = len(matrix1), len(matrix2[0]), len(matrix1[0])
matrix3 = [[0 for _ in range(k1)] for _ in range(m)]

for i in range(m):
    for j in range(k1):
        for z in range(k2):
            matrix3[i][j] = matrix1[i][z] * matrix2[z][j]
print(matrix3)




