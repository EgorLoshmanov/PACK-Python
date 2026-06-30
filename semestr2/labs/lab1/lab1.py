import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle


# conv
# layer1
# layer2
# layer3
# layer4
# avgpool
# fc


# сглаживание
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    gaussian_filter = None
    _HAS_SCIPY = False


# Подмодели ResNet50
class ResNetQuery(nn.Module):
    """ResNet до avgpool (без fc). На выходе эмбеддинг [B, 2048]."""
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # до avgpool включительно

    def forward(self, x):
        x = self.features(x)          # [B, 2048, 1, 1]
        x = x.flatten(1)              # [B, 2048]
        return x


class ResNetFeatureMap(nn.Module):
    """ResNet до layer4 (без avgpool и fc). На выходе feature map [B, 2048, H, W]."""
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])  # до layer4

    def forward(self, x):
        return self.features(x)


# Предобработка
def preprocess_for_resnet_pil(img_pil: Image.Image) -> torch.Tensor:
    """PIL -> Tensor [C,H,W] + Normalize (как для ImageNet)."""
    transform = transforms.Compose([  # Compose объединяет несколько преобразований в одну последовательность
        transforms.ToTensor(),  # переводит PIL (H,W,C, 0..255) → torch.Tensor (C,H,W, float32, 0..1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # вычитает среднее значение по каждому каналу (R,G,B)
                             std=[0.229, 0.224, 0.225]),  # и делит на стандартное отклонение (стандартизация под ImageNet)
    ])
    return transform(img_pil)

def normalize_2d_to_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min = float(x.min())  # находим минимальное значение в массиве и приводим к float
    x_max = float(x.max())  # находим максимальное значение в массиве и приводим к float
    return (x - x_min) / (x_max - x_min + eps)  # выполняем min-max нормализацию: сдвигаем минимум в 0 и масштабируем диапазон к [0,1]


def find_peak_point(sim_map_01: np.ndarray, use_centroid: bool = True, q: float = 0.995):
    """
    Ищем точку пика на карте [H,W] в координатах feature-map.
    - если use_centroid=True: берём top (1-q)% и считаем центр масс
    - иначе: argmax
    Возвращает (fy, fx) float.
    """

    if not use_centroid:
        fy, fx = np.unravel_index(np.argmax(sim_map_01), sim_map_01.shape)
        # np.argmax находит индекс максимального значения (в плоском массиве),
        # np.unravel_index переводит его в координаты (y, x) в 2D массиве
        return float(fy), float(fx)

    # вычисляем q-квантиль (например 0.995 = 99.5-й процентиль),
    thr = np.quantile(sim_map_01, q)

    # создаём булеву маску: True там, где значения входят в top (1-q)% самых больших
    mask = sim_map_01 >= thr

    # создаём булеву маску: True там, где значения входят в top (1-q)% самых больших
    ys, xs = np.where(mask)

    if len(xs) == 0:
        fy, fx = np.unravel_index(np.argmax(sim_map_01), sim_map_01.shape)
        return float(fy), float(fx)

    # считаем центр масс: среднее значение по всем y и среднее по всем x
    return float(np.mean(ys)), float(np.mean(xs))


def clamp_box(left, top, right, bottom, W, H):
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(W, int(right))
    bottom = min(H, int(bottom))
    return left, top, right, bottom


def main():
    query_path = "abc.png"      # картинка, где известный динозавр
    target_path = "A_B_C.jpg"   # картинка, где ищем

    # ROI динозавра на query (left, upper, right, lower)
    query_box = (450, 100, 590, 200)

    # Сглаживание heatmap
    gaussian_sigma = 2.0

    # Точка детекции: centroid top-q (стабильнее чем argmax)
    use_centroid = True
    centroid_quantile = 0.995

    # Масштаб рамки относительно ROI
    bbox_scale = 1.0

    # Загрузка изображений
    query_img = Image.open(query_path).convert("RGB")
    target_img = Image.open(target_path).convert("RGB")

    query_roi = query_img.crop(query_box)
    roi_w, roi_h = query_roi.size

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.DEFAULT
    backbone = resnet50(weights=weights)
    backbone.eval().to(device)

    model_query = ResNetQuery(backbone).eval().to(device)
    model_fm = ResNetFeatureMap(backbone).eval().to(device)

    # Эмбеддинг query (avgpool)
    # Resize делаем на PIL (самый совместимый вариант)
    # Превращает ROI динозавра в нормированный 2048-мерный вектор признаков
    query_roi_224 = query_roi.resize((224, 224), Image.BILINEAR)
    query_tensor = preprocess_for_resnet_pil(query_roi_224).unsqueeze(0).to(device)  # [1,3,224,224]

    with torch.no_grad():
        q = model_query(query_tensor)               # [1,2048]
        q = functional.normalize(q, p=2, dim=1)[0]  # [2048] нормируем вектор

    # Feature map target (layer4)
    # Прогоняет всю target-картинку через ResNet и строит
    # тепловую карту того, где объект больше всего похож на query.
    target_tensor = preprocess_for_resnet_pil(target_img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        fm = model_fm(target_tensor)                  # [1,2048,h,w]
        fm = fm[0]                                    # [2048,h,w]
        # Нормируем каждый spatial-вектор по каналам
        fm = functional.normalize(fm, p=2, dim=0)     # нормировка по channel dim для каждого (h,w)

        # Косинусное сходство: dot(fm[:,y,x], q)
        sim = (fm * q[:, None, None]).sum(dim=0)      # [h,w]

    sim_np = sim.detach().float().cpu().numpy()       # [-1..1] примерно
    sim_01 = normalize_2d_to_01(sim_np)

    # Сглаживаем (если scipy есть)
    sim_for_detect = sim_01
    if _HAS_SCIPY and gaussian_sigma > 0:
        sim_for_detect = gaussian_filter(sim_01, sigma=float(gaussian_sigma))

    # Находим точку пика в feature-map координатах
    fy, fx = find_peak_point(sim_for_detect, use_centroid=use_centroid, q=centroid_quantile)

    h, w = sim_for_detect.shape
    scale_x = target_img.width / w
    scale_y = target_img.height / h

    # Переводим в пиксели target_img (через центр клетки feature-map)
    max_x = int((fx + 0.5) * scale_x)
    max_y = int((fy + 0.5) * scale_y)
    max_x = int(np.clip(max_x, 0, target_img.width - 1))
    max_y = int(np.clip(max_y, 0, target_img.height - 1))

    # Heatmap для визуализации (resize до target_img)
    heat_u8 = np.uint8(255 * sim_for_detect)
    heat_resized = np.array(
        Image.fromarray(heat_u8).resize((target_img.width, target_img.height), Image.BILINEAR)
    )

    # Bounding box
    # масштабируем ROI под target относительно размеров изображений.
    sx_img = target_img.width / query_img.width
    sy_img = target_img.height / query_img.height

    box_w = int(roi_w * sx_img * bbox_scale)
    box_h = int(roi_h * sy_img * bbox_scale)

    left = max_x - box_w // 2
    top = max_y - box_h // 2
    right = left + box_w
    bottom = top + box_h

    box_left, box_top, box_right, box_bottom = clamp_box(left, top, right, bottom,
                                                         target_img.width, target_img.height)
    # фактические размеры после clamp (для matplotlib Rectangle)
    box_w2 = box_right - box_left
    box_h2 = box_bottom - box_top

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(query_roi)
    axes[0].set_title("Query Dinosaur (ROI)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(target_img)
    axes[1].imshow(heat_resized, cmap="jet", alpha=0.45)
    rect = Rectangle((box_left, box_top), box_w2, box_h2,
                     linewidth=3, edgecolor="lime", facecolor="none", linestyle="--")
    axes[1].add_patch(rect)
    axes[1].plot(max_x, max_y, "r*", markersize=15, label="Detected point")
    axes[1].legend(loc="best")
    axes[1].set_title("Heatmap with Detection Box", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(heat_resized, cmap="hot")
    rect2 = Rectangle((box_left, box_top), box_w2, box_h2,
                      linewidth=2.5, edgecolor="white", facecolor="none")
    axes[2].add_patch(rect2)
    axes[2].plot(max_x, max_y, "b*", markersize=12)
    axes[2].set_title("Heatmap (hot) + Bounding Box", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("dinosaur_search_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Сохранение target с рамкой (PIL)
    target_with_box = target_img.copy()
    draw = ImageDraw.Draw(target_with_box)
    draw.rectangle([box_left, box_top, box_right, box_bottom], outline="red", width=4)
    draw.rectangle([box_left - 1, box_top - 1, box_right + 1, box_bottom + 1], outline="white", width=1)
    target_with_box.save("dinosaur_detected.png")

    # Логи
    print(f"Detected point (pixel coords): (x={max_x}, y={max_y})")
    print(f"Bounding box: [{box_left}, {box_top}, {box_right}, {box_bottom}]")
    print(f"Box size: {box_w2}×{box_h2} px")
    print(f"SciPy smoothing: {'ON' if _HAS_SCIPY else 'OFF'} (sigma={gaussian_sigma})")


if __name__ == "__main__":
    main()