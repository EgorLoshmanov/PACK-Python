import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class CFG:
    dataset_root: str = "./EuroSAT_RGB"
    ckpt_path: str = "./runs/eurosat_arcface/best.pt"
    out_dir: str = "./runs/eurosat_arcface"
    seed: int = 42

    img_size: int = 224
    batch_size: int = 128

    train_frac: float = 0.8
    tsne_n: int = 500
    pca_dim: int = 50

    perplexity: int = 20
    max_iter: int = 500

    # для Mac лучше так:
    num_workers: int = 0
    pin_memory: bool = False

    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def stratified_split_indices(targets: List[int], train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array(targets)
    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, test_idx = next(sss.split(idx, y))
    return idx[train_idx], idx[test_idx]


class ResNet18Embedder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        m = resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B, 512, 1, 1]
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


@torch.no_grad()
def compute_embeddings(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_emb, all_y = [], []
    for x, y in tqdm(loader, desc="Embeddings"):
        x = x.to(device)
        emb = l2norm(model(x), dim=1)
        all_emb.append(emb.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_emb, axis=0), np.concatenate(all_y, axis=0)


def plot_tsne(emb: np.ndarray, y: np.ndarray, class_names: List[str], out_path: str, cfg: CFG):
    # subsample
    if len(y) > cfg.tsne_n:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(y), size=cfg.tsne_n, replace=False)
        emb = emb[idx]
        y = y[idx]

    emb_pca = PCA(n_components=min(cfg.pca_dim, emb.shape[1]), random_state=cfg.seed).fit_transform(emb)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=cfg.perplexity,
        method="barnes_hut",
        angle=0.7,
        max_iter=cfg.max_iter,
        verbose=2,
        random_state=cfg.seed
    )

    z = tsne.fit_transform(emb_pca)

    plt.figure(figsize=(10, 8))
    for c in range(len(class_names)):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        plt.scatter(z[mask, 0], z[mask, 1], s=8, label=class_names[c], alpha=0.7)

    plt.legend(markerscale=2, fontsize=9)
    plt.title("EuroSAT (test) embeddings t-SNE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    if not os.path.isdir(cfg.dataset_root):
        raise FileNotFoundError(f"Не нашёл датасет: {cfg.dataset_root}")

    if not os.path.isfile(cfg.ckpt_path):
        raise FileNotFoundError(f"Не нашёл чекпойнт: {cfg.ckpt_path}")

    print("Device:", cfg.device)
    print("Dataset:", cfg.dataset_root)
    print("Checkpoint:", cfg.ckpt_path)

    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # важно: один ImageFolder, чтобы порядок файлов и targets был стабильный
    ds_plain = ImageFolder(root=cfg.dataset_root, transform=None)
    class_names = ds_plain.classes
    targets = ds_plain.targets

    ds_all = ImageFolder(root=cfg.dataset_root, transform=tfm)

    train_idx, test_idx = stratified_split_indices(targets, cfg.train_frac, cfg.seed)
    test_ds = Subset(ds_all, test_idx)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    # грузим модель
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model = ResNet18Embedder(emb_dim=128).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    print("Loaded model OK.")

    # считаем эмбеддинги и делаем t-SNE
    emb_test, y_test = compute_embeddings(model, test_loader, cfg.device)

    out_path = os.path.join(cfg.out_dir, "tsne_test.png")
    plot_tsne(emb_test, y_test, class_names, out_path, cfg)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()