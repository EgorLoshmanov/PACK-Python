import argparse
import numpy as np

def method1(arr_real: np.ndarray, arr_synth: np.ndarray, P: float) -> np.ndarray:
    """
    Формирует результирующий массив, выбирая элементы поэлементно
    из реального или синтетического массива с вероятностью P.

    Parameters
    ----------
    arr_real : np.ndarray
        Массив реальных данных длины n.
    arr_synth : np.ndarray
        Массив синтетических данных длины n.
    P : float
        Вероятность выбора синтетического элемента (0 <= P <= 1).

    Returns
    -------
    np.ndarray
        Новый массив длины n, где каждый элемент с вероятностью P
        взят из arr_synth, иначе из arr_real.
    """    
    n = arr_real.size
    mask = np.random.rand(n) < P
    return np.where(mask, arr_synth, arr_real)

def method2(arr_real: np.ndarray, arr_synth: np.ndarray, P: float) -> np.ndarray:
  
    mask = np.random.rand(*arr_real.shape) < P
    return np.select([mask], [arr_synth], default=arr_real)

    # numpy select, apply along acsis, numpy choices

def method3(arr_real: np.ndarray, arr_synth: np.ndarray, P: float) -> np.ndarray:
    """
    Формирует результирующий массив, сначала случайно определяя количество
    синтетических элементов из биномиального распределения Binomial(n, P),
    затем вставляя их на случайные позиции.

    Parameters
    ----------
    arr_real : np.ndarray
        Массив реальных данных длины n.
    arr_synth : np.ndarray
        Массив синтетических данных длины n.
    P : float
        Вероятность выбора синтетики (0 <= P <= 1).

    Returns
    -------
    np.ndarray
        Новый массив длины n, содержащий k элементов из arr_synth
    """
    n = arr_real.size
    k = np.random.binomial(n, P)
    if k == 0:
        return arr_real.copy()
    idx = np.random.choice(n, size=k, replace=False)
    res_arr = arr_real.copy()
    res_arr[idx] = arr_synth[idx]
    return res_arr

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)
    parser.add_argument("P", type=float)
    args = parser.parse_args()

    arr1 = np.loadtxt(args.file1, dtype=int)
    arr2 = np.loadtxt(args.file2, dtype=int)

    res_arr_method1 = method1(arr1, arr2, args.P)
    print(res_arr_method1)
    print()

    res_arr_method2 = method2(arr1, arr2, args.P)
    print(res_arr_method2)
    print()
     
    res_arr_method3 = method3(arr1, arr2, args.P)
    print(res_arr_method3)
    
if __name__ == "__main__":
    main()