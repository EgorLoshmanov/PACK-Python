import numpy as np

def main() -> None:
    matrix = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=float)

    U, S, Vt = np.linalg.svd(matrix)

    # создаём Σ в виде матрицы
    Sigma = np.zeros_like(matrix)
    np.fill_diagonal(Sigma, S)

    print("U =\n", U)
    print("Сингулярные числа (диагональ Σ) =", S)
    print("V^T =\n", Vt)
    print("A:\n", U @ Sigma @ Vt)

if __name__ == "__main__":
    main()