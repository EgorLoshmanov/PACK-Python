def convolution(matrix1: list[list[int]], matrix2: list[list[int]]) -> list[list[int]]:
    """
    Возвращает свёртку матриц

    Parameters
    ----------
    matrix1 : list[list[int]]
        Первая матрица размера nx x ny
    matrix2 : list[list[int]]
        Вторая матрица размера mx x my


    Returns
    -------
    result : list[list[int]]
        Результат свертки matrix1 и matrix2 - matrix3 размера (nx - mx+1) x (ny - my+1)
 
    """
    matrix3 = [[0 for _ in range(len(matrix1[0]) - len(matrix2[0]) + 1)] for _ in range(len(matrix1) - len(matrix2) + 1)]
    for i in range(len(matrix1) - len(matrix2) + 1):
        for j in range(len(matrix1[0]) - len(matrix2[0]) + 1):
            sum_ = 0
            for u in range(len(matrix2)):
                for v in range(len(matrix2[0])):
                    sum_ += matrix1[i + u][j + v] * matrix2[u][v]
            matrix3[i][j] = sum_

    return matrix3

def main():
    with open("input.txt") as f:
        matrix1, matrix2 = [], []
        for line in f:
            if line[0] != '\n':
                row = [int(x) for x in line.split()]
                matrix1.append(row)
            else:
                break

        for line in f:
            row = [int(x) for x in line.split()]
            matrix2.append(row)

        if len(matrix1) >= len(matrix2) and len(matrix1[0]) >= len(matrix2[0]):
            matrix3 = convolution(matrix1, matrix2)
            with open("output.txt", "w") as f:
                for i in matrix3:
                    f.write(" ".join(map(str, i)))
                    f.write('\n')
        else:
            print("Первая матрица не может быть меньше второй хотя бы в одном из направлений!")

if __name__ == "__main__":
    main()