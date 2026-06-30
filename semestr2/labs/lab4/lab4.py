import argparse
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# -----------------------------
# Models
# -----------------------------

class FCAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()

        # ==== encoder ====
        self.encoder = nn.Sequential(
            nn.Flatten(),   # преобразует изображение [1, 28, 28] -> [784]
            nn.Linear(28 * 28, 512),    # полносвязный слой: 784 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),    # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),     # 256 -> 64
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),         # 64 -> latent_dim
        )

        # === decoder ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),        # latent_dim -> 64
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),     # 64 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),    # 256 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 28 * 28),           # 512 -> 784
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 7x7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_encoder(x)
        h = h.view(x.size(0), -1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), 128, 4, 4)
        return self.conv_decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# -----------------------------
# Training / evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, _ = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    for images, _ in loader:
        images = images.to(device)
        recon, _ = model(images)
        loss = criterion(recon, images)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items


def fit_model(model, train_loader, test_loader, device, epochs, lr, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    history = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        history["train"].append(train_loss)
        history["test"].append(test_loss)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | test_loss={test_loss:.6f}"
        )

    return history


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_history(history, title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train loss")
    plt.plot(history["test"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def save_reconstructions(model, loader, device, save_path: Path, n_images: int = 10) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n_images].to(device)
    recon, _ = model(images)

    images = images.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, n_images, figsize=(1.7 * n_images, 4))
    for i in range(n_images):
        axes[0, i].imshow(images[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Orig {i+1}")

        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Recon {i+1}")

    fig.suptitle("Top row: original images | Bottom row: reconstructed images")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def collect_latent(model, loader, device, max_samples: int = 5000):
    model.eval()
    all_z = []
    all_y = []
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        z = model.encode(images).cpu().numpy()
        all_z.append(z)
        all_y.append(labels.numpy())
        seen += len(images)
        if seen >= max_samples:
            break

    latent = np.concatenate(all_z, axis=0)[:max_samples]
    labels = np.concatenate(all_y, axis=0)[:max_samples]
    return latent, labels


def plot_latent_space(latent: np.ndarray, labels: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, s=8, cmap="tab10", alpha=0.75)
    plt.colorbar(scatter, ticks=range(10), label="Class")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def interactive_decoder(model, latent: np.ndarray, labels: np.ndarray, device: torch.device) -> None:
    """
    Interactive window for 2D latent space.

    - left: scatter plot of encoded test images
    - right: decoded image for chosen z1, z2
    - sliders: manual selection of latent coordinates
    - click on scatter: sliders update to the clicked point
    """
    model.eval()

    z1_min, z1_max = float(latent[:, 0].min()), float(latent[:, 0].max())
    z2_min, z2_max = float(latent[:, 1].min()), float(latent[:, 1].max())

    fig = plt.figure(figsize=(12, 6))
    ax_scatter = plt.axes([0.07, 0.18, 0.45, 0.72])
    ax_img = plt.axes([0.62, 0.35, 0.28, 0.45])
    ax_z1 = plt.axes([0.62, 0.18, 0.28, 0.03])
    ax_z2 = plt.axes([0.62, 0.11, 0.28, 0.03])

    sc = ax_scatter.scatter(latent[:, 0], latent[:, 1], c=labels, s=10, cmap="tab10", alpha=0.7)
    fig.colorbar(sc, ax=ax_scatter, ticks=range(10), label="Class")
    ax_scatter.set_title("2D latent space (click a point)")
    ax_scatter.set_xlabel("z1")
    ax_scatter.set_ylabel("z2")
    ax_scatter.grid(True, alpha=0.3)

    z1_0 = (z1_min + z1_max) / 2.0
    z2_0 = (z2_min + z2_max) / 2.0

    slider_z1 = Slider(ax_z1, "z1", z1_min, z1_max, valinit=z1_0)
    slider_z2 = Slider(ax_z2, "z2", z2_min, z2_max, valinit=z2_0)

    point_marker, = ax_scatter.plot([z1_0], [z2_0], marker="x", markersize=12, markeredgewidth=3)

    def decode_current(z1: float, z2: float):
        z = torch.tensor([[z1, z2]], dtype=torch.float32, device=device)
        decoded = model.decode(z).detach().cpu().numpy()[0, 0]
        return decoded

    image_artist = ax_img.imshow(decode_current(z1_0, z2_0), cmap="gray")
    ax_img.set_title("Decoded image")
    ax_img.axis("off")

    def update(_=None):
        z1 = slider_z1.val
        z2 = slider_z2.val
        decoded = decode_current(z1, z2)
        image_artist.set_data(decoded)
        point_marker.set_data([z1], [z2])
        fig.canvas.draw_idle()

    slider_z1.on_changed(update)
    slider_z2.on_changed(update)

    def on_click(event):
        if event.inaxes != ax_scatter or event.xdata is None or event.ydata is None:
            return
        slider_z1.set_val(float(event.xdata))
        slider_z2.set_val(float(event.ydata))

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Autoencoders")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for FashionMNIST")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimension (for task use 2)")
    parser.add_argument(
        "--interactive-model",
        type=str,
        default="conv",
        choices=["fc", "conv"],
        help="Model used for interactive decoder",
    )
    args = parser.parse_args()

    if args.latent_dim != 2:
        print("Warning: for the task requirement of 2D latent space, latent_dim=2 is recommended.")

    set_seed(42)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading FashionMNIST...")
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    # Fully-connected AE
    print("\nTraining fully-connected autoencoder...")
    fc_model = FCAutoencoder(latent_dim=args.latent_dim).to(device)
    fc_history = fit_model(
        model=fc_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        model_name="FC-AE",
    )
    plot_history(fc_history, "Fully-connected Autoencoder: training curve", output_dir / "fc_training_curve.png")
    save_reconstructions(fc_model, test_loader, device, output_dir / "fc_reconstructions.png")
    fc_latent, fc_labels = collect_latent(fc_model, test_loader, device)
    plot_latent_space(fc_latent, fc_labels, output_dir / "fc_latent_space.png", "FC Autoencoder: 2D latent space")

    # Convolutional AE
    print("\nTraining convolutional autoencoder...")
    conv_model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    conv_history = fit_model(
        model=conv_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        model_name="Conv-AE",
    )
    plot_history(conv_history, "Convolutional Autoencoder: training curve", output_dir / "conv_training_curve.png")
    save_reconstructions(conv_model, test_loader, device, output_dir / "conv_reconstructions.png")
    conv_latent, conv_labels = collect_latent(conv_model, test_loader, device)
    plot_latent_space(conv_latent, conv_labels, output_dir / "conv_latent_space.png", "Conv Autoencoder: 2D latent space")

    # Save weights
    torch.save(fc_model.state_dict(), output_dir / "fc_autoencoder.pth")
    torch.save(conv_model.state_dict(), output_dir / "conv_autoencoder.pth")

    print("\nSaved files:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path}")

    # Interactive decoder
    print("\nOpening interactive decoder window...")
    if args.interactive_model == "fc":
        interactive_decoder(fc_model, fc_latent, fc_labels, device)
    else:
        interactive_decoder(conv_model, conv_latent, conv_labels, device)


if __name__ == "__main__":
    main()