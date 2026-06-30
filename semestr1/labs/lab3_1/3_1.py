from typing import List

def read_matrix(path: str) -> List[List[int]]:
    """Читает матрицу"""
    with open(path) as f:
        matrix = []
        for line in f:
            row = [int(x) for x in line.split()]
            matrix.append(row)
    return matrix

def add_matrixs(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """Поэлементное сложение матриц"""
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]

def sub_matrixs(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """Поэлементное вычитание матриц"""
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]

def print_matrix(m: List[List[int]]):
    """Вывод матрица в терминал"""
    for row in m:
        print(" ".join(map(str, row)))

class Pupa:
    def __init__(self, name: str = "Pupa"):
        self.name = name
        self.salary = 0

    def take_salary(self, amount: int) -> None:
        self.salary += amount

    def do_work(self, filename1: str, filename2: str) -> None:
        matrix1 = read_matrix(filename1)
        matrix2 = read_matrix(filename2)
        mtrx1_add_mtrx2 = add_matrixs(matrix1, matrix2)
        print_matrix(mtrx1_add_mtrx2)

class Lupa:
    def __init__(self, name: str = "Lupa"):
        self.name = name
        self.salary = 0

    def take_salary(self, amount: int) -> None:
        self.salary += amount

    def do_work(self, filename1: str, filename2: str) -> None:
        matrix1 = read_matrix(filename1)
        matrix2 = read_matrix(filename2)
        mtrx1_sub_mtrx2 = sub_matrixs(matrix1, matrix2)
        print_matrix(mtrx1_sub_mtrx2)

class Accountant():
    def give_salary(self, worker, amount: int) -> None:
        worker.take_salary(amount)

def main() -> None:
    pupa = Pupa()
    pupa.do_work("filename1.txt", "filename2.txt")

    lupa = Lupa()
    lupa.do_work("filename1.txt", "filename2.txt")

if __name__ == "__main__":
    main()
