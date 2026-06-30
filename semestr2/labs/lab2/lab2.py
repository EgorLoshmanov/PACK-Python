import os, glob, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


DATA_ROOT = "orl_faces"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BATCH_SIZE = 64          # батч триплетов: anchor/pos/neg -> 3*B изображений
EPOCHS = 20
LR = 1e-3
EMB_DIM = 128
MARGIN = 0.2

# ----------------- utils: load paths -----------------
def load_orl_paths(root):
    paths, labels = [], []
    for sdir in sorted(glob.glob(os.path.join(root, "s*"))):
        label = int(os.path.basename(sdir)[1:])
        for p in sorted(glob.glob(os.path.join(sdir, "*.pgm"))):
            paths.append(p)
            labels.append(label)
    labels = np.array(labels, dtype=np.int64)
    return paths, labels

paths, labels = load_orl_paths(DATA_ROOT)
if len(paths) == 0:
    raise RuntimeError(f"No images found in {DATA_ROOT}. Expected s1..s40/*.pgm")

print("Images:", len(paths), "Unique IDs:", len(np.unique(labels)))

# ----------------- transforms -----------------
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ORL серый -> 3 канала
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------- Triplet dataset -----------------
class ORLTripletDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

        self.idx_by_label = {}
        for i, y in enumerate(self.labels):
            self.idx_by_label.setdefault(int(y), []).append(i)
        self.all_labels = list(self.idx_by_label.keys())

        # оставим только те классы, где >=2 фото (для positive)
        self.valid_labels = [y for y, idxs in self.idx_by_label.items() if len(idxs) >= 2]
        if len(self.valid_labels) < 2:
            raise RuntimeError("Need at least 2 identities with >=2 images each for triplets.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # anchor
        a_path = self.paths[idx]
        a_label = int(self.labels[idx])

        # positive: другое фото того же человека
        pos_candidates = self.idx_by_label[a_label]
        p_idx = idx
        while p_idx == idx:
            p_idx = random.choice(pos_candidates)
        p_path = self.paths[p_idx]

        # negative: случайный другой человек
        n_label = a_label
        while n_label == a_label:
            n_label = random.choice(self.all_labels)
        n_idx = random.choice(self.idx_by_label[n_label])
        n_path = self.paths[n_idx]

        a_img = self.transform(Image.open(a_path))
        p_img = self.transform(Image.open(p_path))
        n_img = self.transform(Image.open(n_path))

        return a_img, p_img, n_img, a_label

dataset = ORLTripletDataset(paths, labels, tfm)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ----------------- Model: ResNet -> embedding -----------------
class ResNetEmbedder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)  # <- не pretrained
        # выкидываем classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # до avgpool
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.backbone(x)          # [B,512,1,1]
        x = x.flatten(1)              # [B,512]
        x = self.fc(x)                # [B,emb_dim]
        x = F.normalize(x, p=2, dim=1)
        return x

model = ResNetEmbedder(EMB_DIM).to(DEVICE)
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)   # Triplet Loss (L2 distance)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------- Train -----------------
model.train()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    total_loss = 0.0
    for a, p, n, _ in loader:
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

        ea = model(a)
        ep = model(p)
        en = model(n)

        loss = criterion(ea, ep, en)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/len(loader):.4f} | time={time.time()-t0:.1f}s")

# ----------------- Embed all images -----------------
model.eval()
@torch.no_grad()
def embed_all(paths, batch_size=64):
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = [tfm(Image.open(p)) for p in batch_paths]
        x = torch.stack(imgs).to(DEVICE)
        e = model(x).cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0)

X = embed_all(paths, batch_size=64)  # [N, EMB_DIM]
print("Embeddings:", X.shape)

# ----------------- t-SNE -----------------
X_std = StandardScaler().fit_transform(X).astype(np.float32)
X_pca = PCA(n_components=min(30, X_std.shape[1]), random_state=SEED).fit_transform(X_std).astype(np.float32)

tsne = TSNE(
    n_components=2,
    perplexity=20,
    init="pca",
    learning_rate="auto",
    method="exact",
    metric="euclidean",
    max_iter=1000,
    verbose=1,
    random_state=SEED,
)
Z = tsne.fit_transform(X_pca)

plt.figure(figsize=(10, 8), dpi=140)
sc = plt.scatter(Z[:,0], Z[:,1], c=labels, s=18)
plt.title("t-SNE of learned ResNet embeddings (Triplet Loss)")
plt.xlabel("tsne-1"); plt.ylabel("tsne-2")
plt.colorbar(sc, label="person id")
plt.tight_layout()
plt.show()

# ----------------- Pairs report (cosine sim/dist) -----------------
def cosine_sim(a, b):
    return float(np.dot(a, b))  # вектора L2-> dot = cosine sim

def sample_pairs(num_same=6, num_diff=6):
    idx_by_label = {}
    for i, y in enumerate(labels):
        idx_by_label.setdefault(int(y), []).append(i)
    all_labels = list(idx_by_label.keys())

    same_pairs = []
    for _ in range(num_same):
        y = random.choice(all_labels)
        i1, i2 = random.sample(idx_by_label[y], 2)
        same_pairs.append((i1, i2, True))

    diff_pairs = []
    for _ in range(num_diff):
        y1, y2 = random.sample(all_labels, 2)
        i1 = random.choice(idx_by_label[y1])
        i2 = random.choice(idx_by_label[y2])
        diff_pairs.append((i1, i2, False))

    return same_pairs + diff_pairs

pairs = sample_pairs(6, 6)

FACE_SIZE = 220
PAD = 20
TEXT_W = 700
ROW_H = FACE_SIZE + 2*PAD
font = ImageFont.load_default()

rows = len(pairs)
W = TEXT_W + 2*FACE_SIZE + 3*PAD
H = rows * ROW_H + PAD

canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)

for r, (i1, i2, is_same) in enumerate(pairs):
    sim = cosine_sim(X[i1], X[i2])
    dist = 1.0 - sim
    y0 = PAD + r*ROW_H

    line = f"{'SAME' if is_same else 'DIFF'} | sim={sim:.3f} | dist={dist:.3f} | id1={labels[i1]} id2={labels[i2]}"
    draw.text((PAD, y0 + 40), line, fill="black", font=font)

    im1 = Image.open(paths[i1]).convert("L").resize((FACE_SIZE, FACE_SIZE), Image.NEAREST).convert("RGB")
    im2 = Image.open(paths[i2]).convert("L").resize((FACE_SIZE, FACE_SIZE), Image.NEAREST).convert("RGB")

    x1 = TEXT_W + PAD
    x2 = TEXT_W + FACE_SIZE + 2*PAD
    canvas.paste(im1, (x1, y0 + PAD))
    canvas.paste(im2, (x2, y0 + PAD))

out_path = "pairs_grid.png"
canvas.save(out_path)
print("Saved:", out_path)

plt.figure(figsize=(14, rows * 2.2), dpi=150)
plt.imshow(canvas)
plt.axis("off")
plt.show()