import numpy as np

def triangles_print(matrix: np.ndarray) -> np.ndarray:
    """
    Возвращает только те строки из матрицы (n, 3),
    которые являются сторонами треугольника.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, 3)
        Каждая строка = тройка (a, b, c).

    Returns
    -------
    np.ndarray
        Подмножество строк, удовлетворяющих неравенствам треугольника.
    """
    a, b, c = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    mask = (a + b > c) & (a + c > b) & (b + c > a)
    return matrix[mask]

def main() -> None:
     M = np.array([
     [3, 4, 5],   # треугольник
     [1, 2, 3],   # не треугольник
     [5, 10, 25], # не треугольник
     [7, 8, 9]    # треугольник
     ])

     print(triangles_print(M))

if __name__ == "__main__":
     main()