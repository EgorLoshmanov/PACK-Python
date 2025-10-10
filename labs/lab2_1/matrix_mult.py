def prod_matrix(matrix1: list[list[int]], matrix2: list[list[int]]) -> list[list[int]]:
    """
    Возвращает произведение двух матриц

    Parameters
    ----------
    matrix1 : list[list[int]]
        Первая матрица размера m x n
    matrix2 : list[list[int]]
        Вторая матрица размера n x k


    Returns
    -------
    result : list[list[int]]
        Результат перемножения matrix1 и matrix2 - matrix3 размера m x k
    """
    matrix3 = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j]
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

        if len(matrix1[0]) == len(matrix2):
            matrix3 = prod_matrix(matrix1, matrix2)

            with open("output.txt", "w") as f:
                for i in matrix3:
                    f.write(" ".join(map(str, i)))
                    f.write('\n')
        else:
            print("Количество столбцов первой матрицы должно совпадать с количеством строк второй матрицы!")

if __name__ == "__main__":
    main()