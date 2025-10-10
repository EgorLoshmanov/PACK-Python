import cv2
import os
import matplotlib.pyplot as plt

def main() -> None:
    path_images = "/home/egor-dmitrievich/.cache/kagglehub/datasets/vpapenko/nails-segmentation/versions/1/images"
    path_masks = "/home/egor-dmitrievich/.cache/kagglehub/datasets/vpapenko/nails-segmentation/versions/1/nails_segmentation/labels"

    files = os.listdir(path_images)
    for files_name in files:
        img_path = os.path.join(path_images, files_name)
        mask_path = os.path.join(path_masks, files_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        res_img = cv2.hconcat([img, mask])

        cv2.imshow("Image and Mask", res_img)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:   # Esc — выход
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()