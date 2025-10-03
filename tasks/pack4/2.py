import numpy as np

def unique_colors_count(img: np.ndarray) -> int:
    """
    Возвращает количество уникальных цветов в изображении RGB.

    Parameters
    ----------
    img : np.ndarray
        Входное изображение в формате NumPy массива формы (h, w, 3),
        где h — высота, w — ширина, 3 — количество каналов (RGB).
        Тип данных np.uint8 (значения от 0 до 255).

    Returns
    -------
    int
        Количество уникальных цветов в изображении. Один цвет определяется
        как тройка значений [R, G, B].
    """
    pixels = img.reshape(-1, 3)                 
    unique_colors = np.unique(pixels, axis=0)          
    return unique_colors.shape[0]

def main() -> None:
    # Пример использования ф-ции unique_colors_count
    # "картинка" 2x2 пикселя, 3 канала (RGB)
    img = np.array([
        [[255, 0, 0],   [0, 255, 0]],   
        [[0, 0, 255],  [255, 0, 0]],     
    ], dtype=np.uint8)

    print("shape:", img.shape)
    print("уникальных цветов:", unique_colors_count(img))

if __name__ == "__main__":
    main()