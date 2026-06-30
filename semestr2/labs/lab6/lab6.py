import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# Конфиг путей

LAB3_DIR   = os.path.join(os.path.dirname(__file__), "..", "lab3")
CKPT_PATH  = os.path.join(LAB3_DIR, "runs", "eurosat_arcface", "best.pt")
DATA_ROOT  = os.path.join(LAB3_DIR, "EuroSAT_RGB")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 1. Классы из Lab3 (необходимо для загрузки чекпойнта)

def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = l2norm(emb, dim=1)
        W   = l2norm(self.weight, dim=1)
        cosine = F.linear(emb, W).clamp(-1.0, 1.0)
        sine   = torch.sqrt((1.0 - cosine * cosine).clamp_min(0.0))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s


class ResNet18Embedder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        m = resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


# 2. Загрузка модели

ckpt       = torch.load(CKPT_PATH, map_location=DEVICE)
class_names = ckpt["class_names"]
num_classes = len(class_names)
emb_dim     = ckpt["cfg"].get("emb_dim", 128)

model = ResNet18Embedder(emb_dim).to(DEVICE)
head  = ArcMarginProduct(emb_dim, num_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
head.load_state_dict(ckpt["head"])
model.eval()
head.eval()
best_acc_val = ckpt.get('best_acc', '?')
acc_str = f"{best_acc_val:.4f}" if isinstance(best_acc_val, float) else str(best_acc_val)
print(f"Loaded checkpoint | best_acc={acc_str}")
print(f"Classes ({num_classes}): {class_names}")

# 3. Плоский список слоёв ResNet18 для визуализации (9 слоёв)
#
#    backbone индексы (nn.Sequential):
#      0=Conv2d, 1=BN, 2=ReLU, 3=MaxPool,
#      4=layer1(2 блока), 5=layer2(2 блока),
#      6=layer3(2 блока), 7=layer4(2 блока), 8=AvgPool

backbone = model.backbone

VIZ_LAYERS: list[tuple[str, nn.Module]] = [
    ("init_relu",       backbone[2]),           # после первой свёртки
    ("layer1_block0",   backbone[4][0]),         # BasicBlock
    ("layer1_block1",   backbone[4][1]),
    ("layer2_block0",   backbone[5][0]),
    ("layer2_block1",   backbone[5][1]),
    ("layer3_block0",   backbone[6][0]),
    ("layer3_block1",   backbone[6][1]),
    ("layer4_block0",   backbone[7][0]),
    ("layer4_block1",   backbone[7][1]),
]
# Индексируем: VIZ_LAYERS[i] = (name, module)
print("\nViz layers:")
for i, (name, _) in enumerate(VIZ_LAYERS):
    print(f"  [{i}] {name}")

# 4. SaveFeatures hook - захват выхода слоя

class SaveFeatures:
    def __init__(self, module: nn.Module):
        self.hook     = module.register_forward_hook(self._hook_fn)
        self.features = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.features = output
        else:
            self.features = output[0]

    def close(self):
        self.hook.remove()


# 5. Вспомогательные трансформации (ImageNet-статистика)

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

normalize   = transforms.Normalize(_MEAN.tolist(), _STD.tolist())
denormalize = transforms.Normalize((-_MEAN / _STD).tolist(), (1.0 / _STD).tolist())

img2tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# 6. Максимизация активации (activation maximization)

def optimize_img(
    layer_module: nn.Module,
    filter_idx: int,
    img_np: np.ndarray,   # float32 HxWx3, [0,1]
    lr: float = 0.05,
    steps: int = 30,
) -> np.ndarray:
    """Один цикл оптимизации: возвращает улучшенное изображение [0,1]."""
    h, w = img_np.shape[:2]
    t = torch.from_numpy(img_np.transpose(2, 0, 1).astype(np.float32))
    t = normalize(t).unsqueeze(0).to(DEVICE).requires_grad_(True)

    activation = SaveFeatures(layer_module)
    opt = torch.optim.Adam([t], lr=lr)  # weight_decay убран — иначе пиксели уходят в нормализованный ноль

    model.eval()
    for _ in range(steps):
        opt.zero_grad()
        model(t)
        # features: [1, C, H, W] — максимизируем среднюю активацию фильтра
        loss = -activation.features[0, filter_idx].mean()
        loss.backward()
        opt.step()

    activation.close()

    out = denormalize(t.detach()[0].cpu()).numpy().transpose(1, 2, 0)
    return np.clip(out, 0, 1)


def generate_max_activation_img(
    layer_idx: int,
    filter_idx: int,
    size: int = 56,
    upscaling_steps: int = 10,
    upscaling_factor: float = 1.2,
    lr: float = 0.05,
    opt_steps: int = 30,
    blur: int = 3,
) -> np.ndarray:
    """
    Генерирует изображение, максимально активирующее filter_idx
    в слое VIZ_LAYERS[layer_idx].
    """
    name, layer_module = VIZ_LAYERS[layer_idx]
    print(f"  Generating for layer [{layer_idx}] {name}, filter {filter_idx}...", end="")

    img = np.random.uniform(100, 160, (size, size, 3)).astype(np.float32) / 255.0

    for step in range(upscaling_steps):
        img = optimize_img(layer_module, filter_idx, img, lr=lr, steps=opt_steps)
        if step < upscaling_steps - 1:
            new_size = int(size * upscaling_factor)
            img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            if blur:
                img = cv2.blur(img, (blur, blur))
            size = new_size
        img = np.clip(img, 0, 1)

    print(" done")
    return img


# 7. Получение средних активаций по всем слоям для одного изображения

@torch.no_grad()
def get_mean_activations(img_tensor: torch.Tensor) -> dict[int, list[float]]:
    """
    Прогоняет img_tensor через модель с хуками на всех VIZ_LAYERS.
    Возвращает {layer_idx: [mean_act_per_filter]}.
    """
    hooks = {i: SaveFeatures(mod) for i, (_, mod) in enumerate(VIZ_LAYERS)}
    model.eval()
    model(img_tensor.to(DEVICE))
    mean_acts = {}
    for i, h in hooks.items():
        f = h.features  # [1, C, H, W]  или [1, C] для init_relu
        n_filters = f.shape[1]
        mean_acts[i] = [f[0, fi].mean().item() for fi in range(n_filters)]
        h.close()
    return mean_acts


# 8. Построение графиков активаций

def plot_activation_bars(
    mean_acts: dict[int, list[float]],
    top_filters: dict[int, list[int]],
    title: str,
    save_path: str,
):
    n = len(mean_acts)
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 2.5))
    axes = axes.flatten()

    for ax_idx, layer_idx in enumerate(sorted(mean_acts.keys())):
        acts = mean_acts[layer_idx]
        tops = top_filters[layer_idx]
        ax = axes[ax_idx]
        ax.bar(range(len(acts)), acts, color="steelblue", alpha=0.7)
        for fi in tops:
            ax.axvline(x=fi, color="grey", linestyle="--", linewidth=0.8)
        ax.plot(tops, [acts[fi] for fi in tops], "ro", markersize=5)
        ax.set_xlabel("filter index")
        ax.set_ylabel("mean activation")
        layer_name = VIZ_LAYERS[layer_idx][0]
        ax.set_title(f"[{layer_idx}] {layer_name}  |  top: {tops}")
        ax.set_xlim(0, len(acts))

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# 9. Построение сетки best-input изображений

def plot_best_inputs(
    fms: dict[int, list[np.ndarray]],
    top_filters: dict[int, list[int]],
    title: str,
    save_path: str,
):
    n_layers = len(fms)
    TOP = max(len(v) for v in top_filters.values())
    fig, axes = plt.subplots(n_layers, TOP, figsize=(TOP * 3, n_layers * 3))

    for row, layer_idx in enumerate(sorted(fms.keys())):
        layer_name = VIZ_LAYERS[layer_idx][0]
        for col, (img, fi) in enumerate(zip(fms[layer_idx], top_filters[layer_idx])):
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"L{layer_idx} F{fi}", fontsize=8)
        # подпись строки
        axes[row][0].set_ylabel(layer_name, fontsize=8, rotation=0, labelpad=60, va="center")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# 10. Полный анализ для одного класса

def analyze_class(
    class_name: str,
    top_k: int = 3,
):
    """
    Для одного изображения из класса class_name:
      - рисует графики активаций по всем слоям,
      - генерирует best-input для топ-K фильтров каждого слоя.
    """
    print(f"\n{'='*60}")
    print(f"  Analyzing class: {class_name}")
    print(f"{'='*60}")

    # Загружаем первое изображение класса
    class_dir = os.path.join(DATA_ROOT, class_name)
    img_file  = sorted(os.listdir(class_dir))[0]
    img_path  = os.path.join(class_dir, img_file)

    pil_img = Image.open(img_path).convert("RGB")
    inp     = img2tensor(pil_img).unsqueeze(0)

    # Сохраняем оригинал
    orig_path = os.path.join(OUT_DIR, f"{class_name}_original.jpg")
    pil_img.save(orig_path)
    print(f"  Original image: {img_path}")

    # Средние активации по всем слоям
    print("  Computing activations...")
    mean_acts = get_mean_activations(inp)

    # Топ-K фильтров на каждом слое
    top_filters: dict[int, list[int]] = {}
    for layer_idx, acts in mean_acts.items():
        sorted_filters = sorted(range(len(acts)), key=lambda fi: acts[fi], reverse=True)
        top_filters[layer_idx] = sorted_filters[:top_k]

    # График активаций
    bar_path = os.path.join(OUT_DIR, f"{class_name}_activations.png")
    print("  Plotting activation bars...")
    plot_activation_bars(
        mean_acts, top_filters,
        title=f"Mean filter activations — {class_name} ({img_file})",
        save_path=bar_path,
    )

    # Генерация best-input для топ-фильтров каждого слоя
    print("  Generating max-activation images...")
    fms: dict[int, list[np.ndarray]] = {}
    for layer_idx in sorted(mean_acts.keys()):
        fms[layer_idx] = []
        for fi in top_filters[layer_idx]:
            img = generate_max_activation_img(
                layer_idx=layer_idx,
                filter_idx=fi,
                size=56,
                upscaling_steps=8,
                upscaling_factor=1.2,
                lr=0.05,
                opt_steps=25,
                blur=3,
            )
            fms[layer_idx].append(img)

    best_path = os.path.join(OUT_DIR, f"{class_name}_best_inputs.png")
    plot_best_inputs(
        fms, top_filters,
        title=f"Max-activation inputs — {class_name}",
        save_path=best_path,
    )


# 11. ЗАДАЧА 2 — один глубокий фильтр, одно "привлекательное" изображение

print("\n" + "="*60)
print("TASK 2: Single deep-layer filter — activation maximization")
print("="*60)

# Берём самый глубокий слой (layer4_block1, индекс 8)
# и автоматически выбираем наиболее активный фильтр на реальном изображении,
# чтобы гарантировать нетривиальный градиент.
SINGLE_LAYER = 8   # layer4_block1

sample_river_path = os.path.join(DATA_ROOT, "River", sorted(os.listdir(os.path.join(DATA_ROOT, "River")))[0])
sample_inp = img2tensor(Image.open(sample_river_path).convert("RGB")).unsqueeze(0)
print(f"  Picking top filter via real image: {sample_river_path}")
_acts = get_mean_activations(sample_inp)
SINGLE_FILTER = int(np.argmax(_acts[SINGLE_LAYER]))
print(f"  Top filter for layer [{SINGLE_LAYER}] {VIZ_LAYERS[SINGLE_LAYER][0]}: filter {SINGLE_FILTER}")

single_img = generate_max_activation_img(
    layer_idx=SINGLE_LAYER,
    filter_idx=SINGLE_FILTER,
    size=56,
    upscaling_steps=12,
    upscaling_factor=1.2,
    lr=0.15,        # выше lr — глубокий слой, малый градиент
    opt_steps=60,
    blur=3,
)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(single_img)
layer_name = VIZ_LAYERS[SINGLE_LAYER][0]
ax.set_title(f"Max-activation image\nLayer [{SINGLE_LAYER}] {layer_name}, filter {SINGLE_FILTER}")
ax.axis("off")
single_path = os.path.join(OUT_DIR, f"task2_single_filter_{layer_name}_f{SINGLE_FILTER}.png")
plt.tight_layout()
plt.savefig(single_path, dpi=150)
plt.close()
print(f"Saved: {single_path}")

# 12. ЗАДАЧА 3 — анализ всех слоёв для "River" и "Highway"

print("\n" + "="*60)
print("TASK 3: Full layer analysis — River & Highway")
print("="*60)

analyze_class("River",   top_k=3)
analyze_class("Highway", top_k=3)

print("\nAll results saved to:", OUT_DIR)
