import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns which weights to prune via learnable
    gate_scores. Gates are obtained by applying Sigmoid to gate_scores,
    then multiplied element-wise with weights before the linear operation.

    During training, the L1 sparsity loss on gates drives many gates toward 0,
    effectively removing those weights from the network.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard learnable weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialized to 0 → sigmoid(0) = 0.5, so all gates start "half-open"
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialize weights with Kaiming uniform (good default for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert raw gate scores to [0, 1] using Sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply: this is differentiable w.r.t. both weight and gate_scores
        pruned_weights = self.weight * gates

        # Standard linear operation: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached from graph, for analysis)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values for this layer."""
        return torch.sigmoid(self.gate_scores).sum()


class SelfPruningNet(nn.Module):
    """
    A 3-hidden-layer feed-forward network for CIFAR-10 (10-class classification).
    Input: 32×32×3 = 3072 features (flattened)
    Hidden layers use PrunableLinear + BatchNorm + ReLU
    Output: 10 logits
    """

    def __init__(self):
        super().__init__()
        self.prunable_layers = nn.ModuleList([
            PrunableLinear(3072, 1024),
            PrunableLinear(1024, 512),
            PrunableLinear(512, 256),
        ])
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                          # Flatten CIFAR images
        x = F.relu(self.bn1(self.prunable_layers[0](x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.prunable_layers[1](x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.prunable_layers[2](x)))
        x = self.output(x)
        return x

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum L1 norms of all gate tensors across all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers)

    def get_all_gates(self) -> torch.Tensor:
        """Concatenate all gate values from all prunable layers into a flat tensor."""
        return torch.cat([layer.get_gates().flatten() for layer in self.prunable_layers])

    def compute_sparsity_level(self, threshold: float = 1e-2) -> float:
        """Percentage of weights whose gate value is below `threshold`."""
        all_gates = self.get_all_gates()
        pruned = (all_gates < threshold).sum().item()
        return 100.0 * pruned / all_gates.numel()


def get_cifar10_loaders(batch_size: int = 256):
    """Download CIFAR-10 and return train/test DataLoaders with standard augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, lam, device):
    """Run a single training epoch; return (avg_total_loss, avg_cls_loss, avg_sparse_loss)."""
    model.train()
    total_loss_sum = cls_loss_sum = sparse_loss_sum = 0.0
    n_batches = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss    = criterion(logits, labels)
        sparse_loss = model.total_sparsity_loss()
        total_loss  = cls_loss + lam * sparse_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum  += total_loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()

    return (total_loss_sum / n_batches,
            cls_loss_sum / n_batches,
            sparse_loss_sum / n_batches)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy (%) on the provided DataLoader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def run_experiment(lam: float, epochs: int, train_loader, test_loader,
                   device, seed: int = 42) -> dict:
    """Train a SelfPruningNet with a given lambda and return result metrics."""
    torch.manual_seed(seed)
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*60}")

    history = {"total": [], "cls": [], "sparse": []}

    for epoch in range(1, epochs + 1):
        t_loss, c_loss, s_loss = train_one_epoch(model, train_loader, optimizer,
                                                  criterion, lam, device)
        scheduler.step()
        history["total"].append(t_loss)
        history["cls"].append(c_loss)
        history["sparse"].append(s_loss)

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sparsity = model.compute_sparsity_level()
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Total: {t_loss:.4f} | CE: {c_loss:.4f} | "
                  f"Sparsity Loss: {s_loss:.1f} | "
                  f"Test Acc: {acc:.2f}% | Gates pruned: {sparsity:.1f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sparsity = model.compute_sparsity_level()
    all_gates = model.get_all_gates().numpy()

    print(f"\n  ✓ Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ✓ Sparsity Level      : {final_sparsity:.1f}%")

    return {
        "lam": lam,
        "model": model,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates": all_gates,
        "history": history,
    }


def plot_gate_distribution(result: dict, save_path: str):
    """Plot histogram of final gate values for a trained model."""
    gates = result["gates"]
    lam = result["lam"]
    acc = result["accuracy"]
    sparsity = result["sparsity"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=100, color="#2563EB", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Gate Value Distribution  |  λ = {lam}\n"
        f"Test Acc: {acc:.2f}%  |  Pruned: {sparsity:.1f}% of weights",
        fontsize=13
    )
    ax.axvline(x=0.01, color="#DC2626", linestyle="--", linewidth=1.5, label="Prune threshold (0.01)")
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved gate distribution plot → {save_path}")


def plot_comparison(results: list, save_path: str):
    """Side-by-side gate distributions for all lambda values."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    for ax, res, color in zip(axes, results, colors):
        ax.hist(res["gates"], bins=100, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.3)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"λ = {res['lam']}\nAcc: {res['accuracy']:.2f}% | Pruned: {res['sparsity']:.1f}%",
                     fontsize=11)
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_yscale("log")

    fig.suptitle("Gate Distributions Across λ Values", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved comparison plot → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS = 30          # increase to 50+ for best accuracy; 30 is fine for demonstrating pruning
    BATCH_SIZE = 256
    LAMBDAS = [1e-5, 1e-4, 1e-3]   # low, medium, high sparsity pressure

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    os.makedirs("outputs", exist_ok=True)
    results = []

    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, train_loader, test_loader, device)
        results.append(res)
        plot_gate_distribution(res, f"outputs/gates_lambda_{lam}.png")

    plot_comparison(results, "outputs/gates_comparison.png")

    # Summary Table
    print("\n" + "="*55)
    print(f"{'λ':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print("-"*55)
    for r in results:
        print(f"{r['lam']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>15.1f}%")
    print("="*55)

    # Save best model (highest accuracy)
    best = max(results, key=lambda r: r["accuracy"])
    torch.save(best["model"].state_dict(), "outputs/best_model.pt")
    print(f"\nBest model (λ={best['lam']}) saved to outputs/best_model.pt")


if __name__ == "__main__":
    main()
