import os
import cv2
import numpy as np
import random

class Generator:
    def __init__(self, image_dir, mask_dir, batch_size=4, res_size=(400, 400)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.res_size = res_size
        self._cursor = 0
        
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')]
        
        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Количество изображений и масок не совпадает")
        
        self.indices = list(range(len(self.image_files)))
        self.on_epoch_end()

    def on_epoch_end(self):
        # перетасовка индексов для случайного порядка выборки в новой "эпохе"
        random.shuffle(self.indices)

    def apply_augmentation(self, image, mask):
        if np.random.random() > 0.5:
            # 1. Поворот на случайный угол в диапазоне [-30°, 30°]
            angle = np.random.uniform(-30, 30)
            # матрица поворота вокруг центра изображения (cx, cy), масштаб = 1.0
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            # применяем поворот к изображению 
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            # к маске - то же преобразование, но с INTER_NEAREST 
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

        if np.random.random() > 0.5:
            # 2. Отражения (горизонтальное и/или вертикальное) 
            if np.random.random() > 0.5:
                # горизонтальный флип (flipCode=1)
                image = cv2.flip(image, 1)  
                mask = cv2.flip(mask, 1)
            if np.random.random() > 0.5:
                # вертикальный флип (flipCode=0)
                image = cv2.flip(image, 0)  
                mask = cv2.flip(mask, 0)
        
        if np.random.random() > 0.5:
            # 3. Случайный кроп (вырезаем кусок и используем его как новое изображение)
            h, w = image.shape[:2]
            # доли от исходного размера (70%..100%)
            crop_h = int(h * np.random.uniform(0.7, 1.0))
            crop_w = int(w * np.random.uniform(0.7, 1.0))
            # случайная верхняя-левая точка так, чтобы кроп целиком помещался внутрь
            top = np.random.randint(0, h - crop_h)
            left = np.random.randint(0, w - crop_w)
            # одинаковый кроп для изображения и маски
            image = image[top:top + crop_h, left:left + crop_w]
            mask = mask[top:top + crop_h, left:left + crop_w]
        
        if np.random.random() > 0.5:
            # 4. Небольшое размытие Гаусса только для изображения 
            image = cv2.GaussianBlur(image, (3, 3), 0)  

        # Приводим изображение и маску к единому целевому размеру (W, H)
        image = cv2.resize(image, self.res_size)
        mask = cv2.resize(mask, self.res_size, interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __iter__(self):
        self.on_epoch_end()
        self._cursor = 0
        return self

    def __next__(self):
        # если дошли до конца - останавливаем итерацию
        if self._cursor >= len(self.indices):
            raise StopIteration

        start = self._cursor
        end = min(self._cursor + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        self._cursor = end  # двигаем курсор

        images, masks = [], []
        for i in batch_indices:
            img = cv2.imread(self.image_files[i])
            mask = cv2.imread(self.mask_files[i], cv2.IMREAD_GRAYSCALE)
            img, mask = self.apply_augmentation(img, mask)
            images.append(img)
            masks.append(mask)

        return np.asarray(images), np.asarray(masks)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))


def main():
    image_dir = "/home/egor-dmitrievich/.cache/kagglehub/datasets/vpapenko/nails-segmentation/versions/1/images"
    mask_dir  = "/home/egor-dmitrievich/.cache/kagglehub/datasets/vpapenko/nails-segmentation/versions/1/nails_segmentation/labels"

    generator = Generator(image_dir, mask_dir, batch_size=4, res_size=(400, 400))

    epoch = 1
    while True:                          
        iter(generator)                  
        print(f"Epoch {epoch}")

        while True:                      
            try:
                imgs, masks = next(generator)   
            except StopIteration:
                break                    # батчи в эпохе закончились

            # возьмём первую пару для показа
            img, msk = imgs[0], masks[0]
            if msk.dtype != np.uint8:
                msk = msk.astype(np.uint8)

            cv2.imshow("image", img)
            cv2.imshow("mask",  msk)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:                # ESC
                cv2.destroyAllWindows()
                return
            if key in (ord('e'), ord('E')):
                break                    # выходим из внутреннего цикла -> новая эпоха

        epoch += 1


if __name__ == "__main__":
    main()