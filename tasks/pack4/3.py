import numpy as np

def main() -> None:
    arr = np.loadtxt("input.txt", dtype=int)
    k = int(input("Введите размер окна: "))

    # (v * kernel)[i] = ∑​v[i+j] * kernel[j]
    moving_avg = np.convolve(arr, np.ones(k) / k, mode='valid')
    print(f"Плавающее среднее вектора: {moving_avg}")
    
if __name__ == "__main__":
    main()