import numpy as np

def sort_by_frequency(arr: np.ndarray) -> np.ndarray:
    """
    Сортирует массив по возрастанию частоты встречаемости значений.
    При равной частоте порядок элементов определяется по самим значениям.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray
        Новый массив той же длины, элементы отсортированы по частоте.
    """
    values, counts = np.unique(arr, return_counts=True)

    # Для каждого элемента arr находим его частоту
    freq = counts[np.searchsorted(values, arr)]

    # Используем np.lexsort: сортируем по (значение, частота)
    idx = np.lexsort((arr, freq))

    return arr[idx]

def main() -> None:
    arr = np.loadtxt("input.txt", dtype=int)
    print(sort_by_frequency(arr))

if __name__ == "__main__":
    main()