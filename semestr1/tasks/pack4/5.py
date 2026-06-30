import numpy as np

def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решает систему линейных уравнений Ax = b методом Гаусса 

    Parameters
    ----------
    A : np.ndarray
        Квадратная матрица коэффициентов системы (размер n x n).
    b : np.ndarray
        Вектор правых частей (размер n или n x 1).

    Returns
    -------
    sol : np.ndarray
        Решение системы уравнений в виде одномерного массива длины n.
    """
    A = A.astype(float)
    b = b.astype(float).reshape(-1, 1)

    # расширенная матрица [A | b]
    M = np.hstack([A, b])
    n = len(b)

    # прямой ход метода Гаусса:
    # 1) нормализация строки i: M[i,:] = M[i,:] / M[i,i]
    #    (на диагонали ставим 1)
    # 2) зануление под диагональю:
    #    M[j,:] = M[j,:] - M[j,i] * M[i,:],  j = i+1..n-1
    for i in range(n):
        M[i] = M[i] / M[i, i]
        for j in range(i + 1, n):
            M[j] = M[j] - M[j, i] * M[i]

    # обратный ход метода Гаусса:
    # для i = n-1..0:
    #   x_i = M[i, -1] - sum_{k=i+1}^{n-1} M[i, k] * x_k
    sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sol[i] = M[i, -1] - np.dot(M[i, i+1:n], sol[i+1:n])
    return sol

# === Пример ===
A = np.array([[3, 4, 2],
              [5, 2, 3],
              [4, 3, 2]])
b = np.array([17, 23, 19])

solution = gauss(A, b)
print("Решение:", solution)
