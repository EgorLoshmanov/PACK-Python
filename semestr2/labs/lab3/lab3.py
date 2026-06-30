import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    dataset_root: str = "./EuroSAT_RGB"
    out_dir: str = "./runs/eurosat_arcface"
    ckpt_name: str = "best.pt"

    seed: int = 42

    img_size: int = 224
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4

    num_workers: int = 0
    pin_memory: bool = False

    emb_dim: int = 128
    arc_s: float = 30.0      # scale (умножаем logits)
    arc_m: float = 0.50      # margin (угловой отступ)
    train_frac: float = 0.8

    pair_vis_n: int = 12

    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Если True — обучение пропускаем и просто грузим чекпойнт
    skip_train: bool = True


# -----------------------------
# Utils
# -----------------------------

# Фиксируем сиды, чтобы результаты (split/рандом) были повторяемыми
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Создаём папку, если её нет
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# L2-нормализация векторов (делаем длину вектора = 1 для cosine-метрик/ArcFace)
def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


# Стратифицированный split индексов: сохраняем пропорции классов в train/test
def stratified_split_indices(targets: List[int], train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array(targets)
    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, test_idx = next(sss.split(idx, y))
    return idx[train_idx], idx[test_idx]


# -----------------------------
# ArcFace head
# -----------------------------
class ArcMarginProduct(nn.Module):
    """
    ArcFace: cos(theta + m) для правильного класса, cos(theta) для остальных, потом масштаб s.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m

        # Веса классов (аналог Linear слоя), будем сравнивать с эмбеддингом по косинусу
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Предвычисляем константы для формулы cos(theta+m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Нормируем и эмбеддинги, и веса классов => работаем с cos(theta)
        emb = l2norm(emb, dim=1)
        W = l2norm(self.weight, dim=1)

        # cos(theta) для всех классов
        cosine = F.linear(emb, W).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(0.0))

        # cos(theta+m) только для target-класса
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot маска target-класса, чтобы применить phi только к нему
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


# -----------------------------
# Model: backbone -> embedding
# -----------------------------
class ResNet18Embedder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        # Берём предобученный ResNet18 как извлекатель признаков
        m = resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B, 512, 1, 1]

        # Проекция 512 -> emb_dim (эмбеддинг, который мы учим через ArcFace)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        return x


# -----------------------------
# Train / Eval
# -----------------------------

# Одна эпоха обучения: считаем loss (ArcFace+CE) и делаем шаг оптимизатора
def train_one_epoch(model, head, loader, opt, device) -> float:
    model.train()
    head.train()

    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        emb = model(x)
        logits = head(emb, y)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


# Оценка точности: классифицируем по cos(theta) (без margin), чтобы честно мерить accuracy
@torch.no_grad()
def eval_accuracy(model, head, loader, device) -> float:
    model.eval()
    head.eval()

    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        emb = model(x)
        emb_n = l2norm(emb, dim=1)
        W = l2norm(head.weight, dim=1)

        logits = F.linear(emb_n, W) * head.s
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# Считаем эмбеддинги для всего датасета
@torch.no_grad()
def compute_embeddings(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_emb, all_y = [], []
    for x, y in tqdm(loader, desc="Embeddings", leave=False):
        x = x.to(device)
        emb = l2norm(model(x), dim=1)
        all_emb.append(emb.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_emb, axis=0), np.concatenate(all_y, axis=0)


# -----------------------------
# Pair inference visualization (pairs + cosine distance)
# -----------------------------

# Рисуем пары картинок из test и их cosine distance, сохраняем png
@torch.no_grad()
def visualize_pairs(dataset, model, device, out_path: str, n_pairs: int = 12):
    model.eval()

    take = min(2 * n_pairs, len(dataset))
    idxs = random.sample(range(len(dataset)), k=take)

    imgs, labels = [], []
    for i in idxs:
        x, y = dataset[i]
        imgs.append(x)
        labels.append(int(y))

    batch = torch.stack(imgs, dim=0).to(device)
    embs = l2norm(model(batch), dim=1).cpu()

    # Cosine distance = 1 - cosine_similarity
    def cos_dist(a, b) -> float:
        return float(1.0 - torch.sum(a * b).item())

    # Собираем пары: часть same-class, часть random
    pairs = []
    by_class: Dict[int, List[int]] = {}
    for k, y in enumerate(labels):
        by_class.setdefault(y, []).append(k)

    for _, arr in by_class.items():
        if len(arr) >= 2 and len(pairs) < n_pairs // 2:
            i1, i2 = random.sample(arr, 2)
            pairs.append((i1, i2))

    while len(pairs) < n_pairs:
        i1, i2 = random.sample(range(len(labels)), 2)
        pairs.append((i1, i2))

    # “Разнормализация” для корректного отображения RGB-картинок
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    cols = 2
    rows = len(pairs)
    plt.figure(figsize=(cols * 5.5, rows * 3.0))

    for r, (i1, i2) in enumerate(pairs, start=1):
        for c, idx in enumerate([i1, i2], start=1):
            plt.subplot(rows, cols, (r - 1) * cols + c)
            img = inv_norm(imgs[idx]).clamp(0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.axis("off")
            plt.title(f"y={labels[idx]}")

        d = cos_dist(embs[i1], embs[i2])
        same = (labels[i1] == labels[i2])
        plt.gcf().text(
            0.5, 1 - (r - 0.5) / rows,
            f"pair {r}: cosine distance={d:.3f} | same={same}",
            ha="center", va="center", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Few-shot (3-5 images) classification via prototypes
# -----------------------------

# Строим прототипы классов: средний эмбеддинг для каждого класса (по train)
@torch.no_grad()
def build_class_prototypes(model, loader, device, num_classes: int) -> np.ndarray:
    emb, y = compute_embeddings(model, loader, device)
    protos = np.zeros((num_classes, emb.shape[1]), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)

    for e, c in zip(emb, y):
        protos[c] += e
        counts[c] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c]

    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    return protos


# Классифицируем набор картинок (3–5 штук): усредняем их эмбеддинги и ищем ближайший прототип
@torch.no_grad()
def classify_images_by_embeddings(
    model: nn.Module,
    image_paths: List[str],
    tfm: transforms.Compose,
    prototypes: np.ndarray,
    class_names: List[str],
    device: str,
    topk: int = 3,
):
    imgs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        imgs.append(tfm(im))

    x = torch.stack(imgs, dim=0).to(device)

    # few-shot: средний эмбеддинг по нескольким примерам
    e = l2norm(model(x), dim=1).mean(dim=0, keepdim=True)
    e = l2norm(e, dim=1).cpu().numpy()[0]

    sims = prototypes @ e  # cosine similarity
    order = np.argsort(-sims)[:topk]
    return [(class_names[i], float(sims[i])) for i in order]


# Сохраняем чекпойнт: веса модели/головы + список классов + лучший accuracy
def save_ckpt(path: str, model: nn.Module, head: nn.Module, class_names: List[str], cfg: CFG, best_acc: float):
    torch.save({
        "model": model.state_dict(),
        "head": head.state_dict(),
        "class_names": class_names,
        "cfg": cfg.__dict__,
        "best_acc": float(best_acc),
    }, path)


# Загружаем чекпойнт: восстанавливаем веса и возвращаем словарь с метаданными
def load_ckpt(path: str, model: nn.Module, head: nn.Module, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    return ckpt


# -----------------------------
# Main
# -----------------------------

def main():
    cfg = CFG()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    if not os.path.isdir(cfg.dataset_root):
        raise FileNotFoundError(
            f"Не нашёл папку датасета: {cfg.dataset_root}\n"
            f"Ожидаю структуру: {cfg.dataset_root}/ClassName/*.jpg"
        )

    print("Device:", cfg.device)
    print("Dataset root:", cfg.dataset_root)

    train_tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Считываем targets
    ds_plain = ImageFolder(root=cfg.dataset_root, transform=None)
    class_names = ds_plain.classes
    num_classes = len(class_names)
    targets = ds_plain.targets

    # Два датасета с разными transform, но одинаковым порядком файлов
    ds_train_all = ImageFolder(root=cfg.dataset_root, transform=train_tfm)
    ds_test_all = ImageFolder(root=cfg.dataset_root, transform=test_tfm)

    train_idx, test_idx = stratified_split_indices(targets, cfg.train_frac, cfg.seed)

    train_ds = Subset(ds_train_all, train_idx)
    test_ds = Subset(ds_test_all, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    model = ResNet18Embedder(cfg.emb_dim).to(cfg.device)
    head = ArcMarginProduct(cfg.emb_dim, num_classes, s=cfg.arc_s, m=cfg.arc_m).to(cfg.device)

    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)

    if cfg.skip_train:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"skip_train=True, но нет чекпойнта: {ckpt_path}")
        ckpt = load_ckpt(ckpt_path, model, head, cfg.device)
        print("Loaded:", ckpt_path, "| best_acc:", ckpt.get("best_acc"))
    else:
        opt = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        best_acc = -1.0
        for epoch in range(1, cfg.epochs + 1):
            loss = train_one_epoch(model, head, train_loader, opt, cfg.device)
            acc = eval_accuracy(model, head, test_loader, cfg.device)
            print(f"Epoch {epoch:02d}/{cfg.epochs} | loss={loss:.4f} | test_acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                save_ckpt(ckpt_path, model, head, class_names, cfg, best_acc)

        print("Best test_acc:", best_acc)
        print("Saved:", ckpt_path)

        # Грузим лучший чекпойнт перед визуализацией/инференсом
        load_ckpt(ckpt_path, model, head, cfg.device)

    # 4) Визуализация пар (2 картинки + distance)
    pair_path = os.path.join(cfg.out_dir, "pairs_distance.png")
    visualize_pairs(test_ds, model, cfg.device, pair_path, n_pairs=cfg.pair_vis_n)
    print("Pairs saved:", pair_path)

    # 5) Few-shot: строим прототипы по train split
    proto_loader = DataLoader(
        Subset(ds_test_all, train_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    prototypes = build_class_prototypes(model, proto_loader, cfg.device, num_classes)

    # Few-shot пример: 3 картинки одного класса в папке ./query/
    query_paths = ["./query/1.jpg", "./query/2.jpg", "./query/3.jpg"]
    top = classify_images_by_embeddings(model, query_paths, test_tfm, prototypes, class_names, cfg.device, topk=3)
    print("Few-shot top3:", top)


if __name__ == "__main__":
    main()