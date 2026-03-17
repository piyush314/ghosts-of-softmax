#!/usr/bin/env python3
"""Natural instability detection on CIFAR-10 with ResNet-18.

Train at several LRs (including one causing mid-training instability).
Log r = τ/ρ_a at every step. Test whether r approaching 1 is a leading
indicator of loss spikes.

Outputs:
  - cache/resnet18_instability_{lr}.pt
  - paper/figures/plots/resnet18-instability-{lr}.{png,pdf}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-exp28")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ghosts.plotting import (
    PALETTE,
    add_end_labels,
    add_subtitle,
    apply_plot_style,
    finish_figure,
    format_percent_axis,
)
from ghosts.reporting import repo_relpath, scalar_stats, write_summary

DATA_DIR = Path(__file__).resolve().parent / "cache"
PLOT_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10 (3x3 stem, no maxpool)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_loaders(data_root: Path, batch_size: int, num_workers: int = 2,
                 download: bool = True, dataset: str = "cifar10",
                 train_samples: int | None = None,
                 test_samples: int | None = None) -> Tuple[DataLoader, DataLoader]:
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    if dataset == "fake":
        train_ds = datasets.FakeData(
            size=train_samples or 128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=tfm_train,
        )
        test_ds = datasets.FakeData(
            size=test_samples or 64,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=tfm_test,
        )
    else:
        train_ds = datasets.CIFAR10(root=str(data_root), train=True, transform=tfm_train,
                                     download=download)
        test_ds = datasets.CIFAR10(root=str(data_root), train=False, transform=tfm_test,
                                    download=download)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def estimate_rho_fd(model: nn.Module, X: torch.Tensor, v: torch.Tensor,
                    eps: float = 1e-4) -> float:
    """Estimate rho_a = pi / Delta_a via finite difference."""
    theta0 = parameters_to_vector(model.parameters()).detach()

    with torch.no_grad():
        z0 = model(X)
        vector_to_parameters(theta0 + eps * v, model.parameters())
        z1 = model(X)
        vector_to_parameters(theta0, model.parameters())

    dlogits = (z1 - z0) / eps
    spread = dlogits.max(dim=1).values - dlogits.min(dim=1).values
    delta_a = float(spread.max().item())
    return math.pi / max(delta_a, 1e-12)


def sgd_effective_grad_vector(opt: torch.optim.Optimizer) -> torch.Tensor:
    """Return the effective SGD update direction before multiplying by lr.

    This includes weight decay and momentum state, matching the direction
    used by ``torch.optim.SGD.step`` for a single upcoming update.
    """
    vecs = []
    device = None

    for group in opt.param_groups:
        momentum = group.get('momentum', 0.0)
        dampening = group.get('dampening', 0.0)
        weight_decay = group.get('weight_decay', 0.0)
        nesterov = group.get('nesterov', False)
        maximize = group.get('maximize', False)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.detach()
            device = grad.device
            if maximize:
                grad = -grad
            if weight_decay != 0:
                grad = grad.add(p.detach(), alpha=weight_decay)

            if momentum != 0:
                state = opt.state[p]
                buf = state.get('momentum_buffer')
                if buf is None:
                    buf = grad
                else:
                    buf = buf.detach().mul(momentum).add(grad, alpha=1 - dampening)
                grad = grad.add(buf, alpha=momentum) if nesterov else buf

            vecs.append(grad.reshape(-1))

    if not vecs:
        return torch.zeros(1, device=device or torch.device('cpu'))
    return torch.cat(vecs)


def train_one_epoch(model: nn.Module, opt: torch.optim.Optimizer,
                    train_loader: DataLoader, device: torch.device,
                    log_every: int = 50, rho_batch_size: int = 256) -> List[Dict]:
    """Train for one epoch, logging r = tau/rho_a at intervals."""
    model.train()
    logs = []

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Compute gradient norm and direction
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=device))
        g = torch.cat(grads)
        g_norm = float(g.norm().item())

        # Get the actual upcoming SGD step scale, including momentum state.
        eff_g = sgd_effective_grad_vector(opt)
        eff_g_norm = float(eff_g.norm().item())
        lr = opt.param_groups[0]['lr']
        tau = lr * eff_g_norm

        # Compute rho_a periodically
        if batch_idx % log_every == 0 and eff_g_norm > 1e-12:
            v = -eff_g / (eff_g_norm + 1e-12)
            # Use subset of current batch for rho estimation
            X_rho = X[:min(rho_batch_size, len(X))]
            rho_a = estimate_rho_fd(model, X_rho, v, eps=1e-4)
            r = tau / rho_a if rho_a > 0 else float('inf')

            acc = (logits.argmax(1) == y).float().mean().item()
            logs.append({
                'batch': batch_idx,
                'loss': loss.item(),
                'acc': acc,
                'g_norm': g_norm,
                'tau': tau,
                'rho_a': rho_a,
                'r': r,
                'lr': lr,
            })

        opt.step()

    return logs


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def train_one_epoch_rho_ctrl(model: nn.Module, opt: torch.optim.Optimizer,
                             train_loader: DataLoader, device: torch.device,
                             log_every: int = 50, rho_batch_size: int = 256,
                             target_r: float = 1.0) -> List[Dict]:
    """Train for one epoch with rho_a controller (lr_eff = target_r * rho_a / ||g||)."""
    model.train()
    logs = []

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Compute gradient norm and direction
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=device))
        g = torch.cat(grads)
        g_norm = float(g.norm().item())

        eff_g = sgd_effective_grad_vector(opt)
        eff_g_norm = float(eff_g.norm().item())

        if eff_g_norm < 1e-12:
            opt.step()
            continue

        # Compute rho_a along the actual upcoming SGD step direction and
        # choose lr_eff so that the true step norm satisfies tau = rho_a.
        v = -eff_g / eff_g_norm
        X_rho = X[:min(rho_batch_size, len(X))]
        rho_a = estimate_rho_fd(model, X_rho, v, eps=1e-4)

        lr_eff = rho_a / eff_g_norm
        for pg in opt.param_groups:
            pg['lr'] = lr_eff

        tau = lr_eff * eff_g_norm
        r = tau / rho_a if rho_a > 0 else 1.0

        if batch_idx % log_every == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            logs.append({
                'batch': batch_idx,
                'loss': loss.item(),
                'acc': acc,
                'g_norm': g_norm,
                'tau': tau,
                'rho_a': rho_a,
                'r': r,
                'lr': lr_eff,
            })

        opt.step()

    return logs


def run_training(lr: float, epochs: int, batch_size: int, device: torch.device,
                 data_root: Path, seed: int, log_every: int = 50,
                 use_rho_ctrl: bool = False, target_r: float = 1.0,
                 dataset: str = "cifar10", train_samples: int | None = None,
                 test_samples: int | None = None, num_workers: int = 2) -> Dict:
    """Run full training at given LR, logging r throughout."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, test_loader = make_loaders(
        data_root,
        batch_size,
        num_workers=num_workers,
        dataset=dataset,
        train_samples=train_samples,
        test_samples=test_samples,
    )
    model = build_cifar_resnet18().to(device)
    # For rho_ctrl mode, start with a dummy LR (will be overwritten each step)
    init_lr = 0.1 if use_rho_ctrl else lr
    opt = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

    all_logs = []
    epoch_logs = []

    for epoch in range(epochs):
        if use_rho_ctrl:
            logs = train_one_epoch_rho_ctrl(model, opt, train_loader, device, log_every,
                                            target_r=target_r)
        else:
            logs = train_one_epoch(model, opt, train_loader, device, log_every)

        # Add epoch info
        for log in logs:
            log['epoch'] = epoch
            log['step'] = epoch * len(train_loader) + log['batch']
        all_logs.extend(logs)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, device)
        epoch_logs.append({
            'epoch': epoch,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_loss': logs[-1]['loss'] if logs else float('nan'),
        })

        print(f"  epoch {epoch}: test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
              f"r_max={max(l['r'] for l in logs):.2f}")

    return {'step_logs': all_logs, 'epoch_logs': epoch_logs, 'lr': lr}


def make_plot(data: Dict, out_png: Path, out_pdf: Path) -> None:
    """Plot loss, r, and their relationship."""
    logs = data['step_logs']
    lr = data['lr']

    steps = np.array([l['step'] for l in logs])
    losses = np.array([l['loss'] for l in logs])
    rs = np.array([l['r'] for l in logs])
    rho_as = np.array([l['rho_a'] for l in logs])

    apply_plot_style(font_size=10, title_size=13, label_size=10, tick_size=9)
    lr_str = f"{lr:g}"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"With LR={lr_str}, the instability signal rises before the loss spike arrives",
        fontsize=13,
        fontweight='bold',
    )
    fig.text(
        0.08,
        0.94,
        "Single-run view of loss, normalized step size, analytic radius, and their relationship over time.",
        fontsize=10,
        color=PALETTE["mid_gray"],
        ha="left",
    )

    # Panel 0: Loss over time
    ax = axes[0, 0]
    ax.semilogy(steps, losses, color=PALETTE['red'], lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss rises as training approaches the unstable regime', loc='left', fontweight='bold')
    add_subtitle(ax, "Training loss on a log scale.", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 1: r = tau/rho_a over time
    ax = axes[0, 1]
    ax.semilogy(steps, rs, color=PALETTE['blue'], lw=1.5)
    ax.axhline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$r = \tau / \rho_a$')
    ax.set_title('The normalized step approaches the stability boundary', loc='left', fontweight='bold')
    add_subtitle(ax, "The critical line is r = 1.", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: rho_a over time
    ax = axes[1, 0]
    ax.semilogy(steps, rho_as, color=PALETTE['green'], lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\rho_a$')
    ax.set_title('The analytic radius shrinks as the model becomes fragile', loc='left', fontweight='bold')
    add_subtitle(ax, "Smaller rho_a means less headroom for the same optimizer step.", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Loss vs r (scatter)
    ax = axes[1, 1]
    thirds = np.array_split(np.arange(len(steps)), 3)
    scatter_specs = [
        (thirds[0], PALETTE["blue"], "early"),
        (thirds[1], PALETTE["gold"], "middle"),
        (thirds[2], PALETTE["red"], "late"),
    ]
    for idx, color, label in scatter_specs:
        ax.scatter(rs[idx], losses[idx], color=color, s=12, alpha=0.55, label=label)
    ax.axvline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r = \tau / \rho_a$')
    ax.set_ylabel('Training Loss')
    ax.set_title('Late training points crowd near the instability boundary', loc='left', fontweight='bold')
    add_subtitle(ax, "Color indicates when the point occurred during training.", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    finish_figure(fig, rect=[0, 0, 1, 0.92])
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def make_comparison_plot(all_data: Dict, out_png: Path, out_pdf: Path) -> None:
    """Plot comparison across LRs."""
    lrs = sorted(all_data.keys())

    apply_plot_style(font_size=10, title_size=12, label_size=10, tick_size=9)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Larger learning rates move ResNet-18 closer to the instability boundary",
        fontsize=13,
        fontweight='bold',
    )
    fig.text(
        0.08,
        0.94,
        "Comparison across fixed learning rates. Right-side labels replace legends so the trajectories remain readable.",
        fontsize=10,
        color=PALETTE["mid_gray"],
        ha="left",
    )

    colors = [PALETTE['blue'], PALETTE['green'], PALETTE['gold'], PALETTE['red']]

    # Panel 0: Loss curves
    ax = axes[0, 0]
    loss_specs = []
    for i, lr in enumerate(lrs):
        logs = all_data[lr]['step_logs']
        steps = [l['step'] for l in logs]
        losses = [l['loss'] for l in logs]
        ax.semilogy(steps, losses, color=colors[i % len(colors)], lw=1.5, alpha=0.8)
        loss_specs.append((losses[-1], f"LR={lr:g}", colors[i % len(colors)], None))
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Higher learning rates produce earlier loss blow-up', loc='left', fontweight='bold')
    add_subtitle(ax, "Each line is a separate fixed-LR training run.", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_end_labels(
        ax,
        steps,
        loss_specs,
        fontsize=8,
    )

    # Panel 1: r curves
    ax = axes[0, 1]
    r_specs = []
    for i, lr in enumerate(lrs):
        logs = all_data[lr]['step_logs']
        steps = [l['step'] for l in logs]
        rs = [l['r'] for l in logs]
        ax.semilogy(steps, rs, color=colors[i % len(colors)], lw=1.5, alpha=0.8)
        r_specs.append((rs[-1], f"LR={lr:g}", colors[i % len(colors)], None))
    ax.axhline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$r = \tau / \rho_a$')
    ax.set_title('The normalized step is the leading instability signal', loc='left', fontweight='bold')
    add_subtitle(ax, "Curves above r = 1 have already outrun the local analytic radius.", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_end_labels(
        ax,
        steps,
        r_specs,
        fontsize=8,
    )

    # Panel 2: Test accuracy
    ax = axes[1, 0]
    acc_specs = []
    for i, lr in enumerate(lrs):
        epoch_logs = all_data[lr]['epoch_logs']
        epochs = [l['epoch'] for l in epoch_logs]
        accs = [l['test_acc'] for l in epoch_logs]
        ax.plot(epochs, accs, color=colors[i % len(colors)], lw=2, marker='o', markersize=4)
        acc_specs.append((accs[-1], f"LR={lr:g}", colors[i % len(colors)], None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy falls first for the most aggressive learning rates', loc='left', fontweight='bold')
    add_subtitle(ax, "Evaluation is shown per epoch on the same test split.", fontsize=9)
    format_percent_axis(ax, xmax=1.0)
    ax.grid(True, alpha=0.3)
    add_end_labels(
        ax,
        epochs,
        acc_specs,
        fontsize=8,
    )

    # Panel 3: Max r per epoch
    ax = axes[1, 1]
    max_r_specs = []
    for i, lr in enumerate(lrs):
        logs = all_data[lr]['step_logs']
        # Group by epoch
        epochs_r = {}
        for l in logs:
            e = l['epoch']
            if e not in epochs_r:
                epochs_r[e] = []
            epochs_r[e].append(l['r'])
        epochs = sorted(epochs_r.keys())
        max_rs = [max(epochs_r[e]) for e in epochs]
        ax.semilogy(epochs, max_rs, color=colors[i % len(colors)], lw=2, marker='o', markersize=4)
        max_r_specs.append((max_rs[-1], f"LR={lr:g}", colors[i % len(colors)], None))
    ax.axhline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Max $r$ per epoch')
    ax.set_title('The per-epoch maximum r separates safe and unsafe runs', loc='left', fontweight='bold')
    add_subtitle(ax, "Values above 1 indicate at least one unsafe step during that epoch.", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_end_labels(
        ax,
        epochs,
        max_r_specs,
        fontsize=8,
    )

    finish_figure(fig, rect=[0, 0, 1, 0.92])
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def trim_epochs(seed_dict: Dict, max_ep: int) -> Dict:
    """Trim epoch_logs and step_logs to max_ep epochs."""
    out = {}
    for s, d in seed_dict.items():
        out[s] = {
            'epoch_logs': d['epoch_logs'][:max_ep],
            'step_logs': [l for l in d['step_logs']
                          if l['epoch'] < max_ep],
            'lr': d.get('lr', 0),
        }
    return out


def load_rho_variants(data_dir: Path,
                      max_ep: int = 10) -> Dict:
    """Load r=0.5, r=1.0, r=2.0, r=4.0 rho_ctrl data."""
    tags = [('', 1.0), ('_r0p5', 0.5), ('_r2p0', 2.0),
            ('_r4p0', 4.0)]
    variants = {}
    for suffix, r_val in tags:
        # Prefer momentum-corrected 5-seed files
        pt = data_dir / f"resnet18_rho_ctrl{suffix}_momentum_5seed.pt"
        if not pt.exists():
            pt = data_dir / f"resnet18_rho_ctrl{suffix}.pt"
        if pt.exists():
            raw = torch.load(pt, weights_only=False)
            variants[r_val] = trim_epochs(raw, max_ep)
    return variants if variants else None


def make_multiseed_plot(all_data: Dict, out_png: Path, out_pdf: Path,
                        rho_variants: Dict = None) -> None:
    """Plot comparison across LRs with multi-seed median+IQR.

    rho_variants: dict mapping target_r -> seed_dict, e.g.
        {0.5: {0: {...}, 1: {...}}, 1.0: {...}, 2.0: {...}}
    """
    lrs = sorted(all_data.keys())
    seeds = sorted(all_data[lrs[0]].keys())

    apply_plot_style(font_size=10, title_size=12, label_size=10, tick_size=9)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        "Across seeds, radius-controlled runs keep ResNet-18 in the stable regime",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.92,
        "Median and IQR for standard fixed learning rates and several rho-controller targets.",
        fontsize=10,
        color=PALETTE["mid_gray"],
        ha="left",
    )

    colors = [PALETTE['blue'], PALETTE['green'],
              PALETTE['gold'], PALETTE['red']]
    ctrl_color = PALETTE['purple']
    ctrl_styles = [
        (0.5, ctrl_color, 'D', r'$\rho_a$-ctrl $r{=}0.5$'),
        (1.0, ctrl_color, 's', r'$\rho_a$-ctrl $r{=}1$'),
        (2.0, ctrl_color, '^', r'$\rho_a$-ctrl $r{=}2$'),
        (4.0, ctrl_color, 'o', r'$\rho_a$-ctrl $r{=}4$'),
    ]

    def stats(arrs):
        arr = np.array(arrs)
        return (np.median(arr, 0), np.percentile(arr, 25, 0),
                np.percentile(arr, 75, 0))

    def plotband(ax, ep, arrs, color, label,
                 log=False, marker='o', ms=3,
                 mfc=None, mec=None, mew=1):
        mkw = dict(marker=marker, markersize=ms,
                   markerfacecolor=mfc or color,
                   markeredgecolor=mec or color,
                   markeredgewidth=mew)
        med, q25, q75 = stats(arrs)
        if log:
            ax.fill_between(ep, np.maximum(q25, 1e-4),
                            np.maximum(q75, 1e-4),
                            alpha=0.15, color=color)
            ax.semilogy(ep, np.maximum(med, 1e-4), color=color,
                        lw=2, label=label, **mkw)
        else:
            ax.fill_between(ep, q25, q75, alpha=0.15, color=color)
            ax.plot(ep, med, color=color, lw=2, label=label,
                    **mkw)

    def maxr_per_epoch(seed_data):
        logs = seed_data['step_logs']
        by_ep = {}
        for l in logs:
            by_ep.setdefault(l['epoch'], []).append(l['r'])
        ep_list = sorted(by_ep.keys())
        return ep_list, [max(by_ep[e]) for e in ep_list]

    # Panel 0: Test loss
    ax = axes[0]
    loss_label_specs = []
    for i, lr in enumerate(lrs):
        loss_all = [[l['test_loss'] for l in all_data[lr][s]['epoch_logs']]
                    for s in seeds]
        plotband(ax, list(range(len(loss_all[0]))), loss_all,
                 colors[i], f'LR={lr}', log=True)
        med, _, _ = stats(loss_all)
        loss_label_specs.append((max(med[-1], 1e-4), f'LR={lr:g}', colors[i], None))
    if rho_variants:
        for tr, col, mk, lab in ctrl_styles:
            if tr not in rho_variants:
                continue
            rd = rho_variants[tr]
            rs = sorted(rd.keys())
            vals = [[l['test_loss'] for l in rd[s]['epoch_logs']]
                    for s in rs]
            plotband(ax, list(range(len(vals[0]))), vals,
                     col, lab, log=True, marker=mk, ms=6,
                     mfc='white', mec=col, mew=1.5)
            med, _, _ = stats(vals)
            loss_label_specs.append((max(med[-1], 1e-4), lab, col, "bold" if tr == 1.0 else None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Radius-controlled runs avoid the worst loss blow-up', loc='left', fontweight='bold')
    add_subtitle(ax, "Median loss across seeds on a log scale.", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_end_labels(ax, list(range(len(loss_all[0]))), loss_label_specs, fontsize=7)

    # Panel 1: Test accuracy
    ax = axes[1]
    acc_label_specs = []
    for i, lr in enumerate(lrs):
        accs = [[l['test_acc'] for l in all_data[lr][s]['epoch_logs']]
                for s in seeds]
        plotband(ax, list(range(len(accs[0]))), accs,
                 colors[i], f'LR={lr}')
        med, _, _ = stats(accs)
        acc_label_specs.append((med[-1], f'LR={lr:g}', colors[i], None))
    if rho_variants:
        for tr, col, mk, lab in ctrl_styles:
            if tr not in rho_variants:
                continue
            rd = rho_variants[tr]
            rs = sorted(rd.keys())
            vals = [[l['test_acc'] for l in rd[s]['epoch_logs']]
                    for s in rs]
            plotband(ax, list(range(len(vals[0]))), vals,
                     col, lab, marker=mk, ms=6,
                     mfc='white', mec=col, mew=1.5)
            med, _, _ = stats(vals)
            acc_label_specs.append((med[-1], lab, col, "bold" if tr == 1.0 else None))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('The best rho targets preserve more accuracy', loc='left', fontweight='bold')
    add_subtitle(ax, "Accuracy is shown as a proportion on the held-out set.", fontsize=9)
    format_percent_axis(ax, xmax=1.0)
    ax.grid(True, alpha=0.3)
    add_end_labels(ax, list(range(len(accs[0]))), acc_label_specs, fontsize=7)

    # Panel 2: Max r per epoch
    ax = axes[2]
    r_label_specs = []
    for i, lr in enumerate(lrs):
        mr_all = []
        for s in seeds:
            ep_list, mrs = maxr_per_epoch(all_data[lr][s])
            mr_all.append(mrs)
        plotband(ax, ep_list, mr_all, colors[i],
                 f'LR={lr}', log=True)
        med, _, _ = stats(mr_all)
        r_label_specs.append((max(med[-1], 1e-4), f'LR={lr:g}', colors[i], None))
    if rho_variants:
        for tr, col, mk, lab in ctrl_styles:
            if tr not in rho_variants:
                continue
            rd = rho_variants[tr]
            rs = sorted(rd.keys())
            mr_all = []
            for s in rs:
                ep_list, mrs = maxr_per_epoch(rd[s])
                mr_all.append(mrs)
            plotband(ax, ep_list, mr_all, col, lab,
                     log=True, marker=mk, ms=6,
                     mfc='white', mec=col, mew=1.5)
            med, _, _ = stats(mr_all)
            r_label_specs.append((max(med[-1], 1e-4), lab, col, "bold" if tr == 1.0 else None))
    ax.axhline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Max $r$ per epoch')
    ax.set_title('Well-tuned rho targets hold the step close to the boundary', loc='left', fontweight='bold')
    add_subtitle(ax, "The dashed line marks the r = 1 stability threshold.", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_end_labels(ax, ep_list, r_label_specs, fontsize=7)

    finish_figure(fig, rect=[0, 0, 1, 0.90])
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def build_standard_summary(all_data: Dict, lrs: List[float], config: Dict,
                           out_pt: Path, out_png: Path, out_pdf: Path,
                           out_summary: Path) -> Dict:
    summary = {
        "experiment_id": "resnetnatural",
        "mode": "fixed-lr-scan",
        "config": config,
        "artifacts": {
            "data": repo_relpath(out_pt, REPO_ROOT),
            "plot_png": repo_relpath(out_png, REPO_ROOT),
            "plot_pdf": repo_relpath(out_pdf, REPO_ROOT),
            "summary_json": repo_relpath(out_summary, REPO_ROOT),
        },
        "learning_rates": {},
    }
    for lr in lrs:
        lr_summary = {
            "final_acc": scalar_stats(
                [all_data[lr][seed]["epoch_logs"][-1]["test_acc"] for seed in all_data[lr]]
            ),
            "max_r": scalar_stats(
                [max(log["r"] for log in all_data[lr][seed]["step_logs"]) for seed in all_data[lr]]
            ),
            "count_r_gt_1": scalar_stats(
                [
                    sum(1 for log in all_data[lr][seed]["step_logs"] if log["r"] > 1.0)
                    for seed in all_data[lr]
                ]
            ),
        }
        summary["learning_rates"][str(lr)] = lr_summary
    return summary


def build_rho_ctrl_summary(all_data: Dict, config: Dict,
                           out_pt: Path, out_summary: Path) -> Dict:
    seeds = sorted(all_data.keys())
    return {
        "experiment_id": "resnetnatural",
        "mode": "rho_ctrl",
        "config": config,
        "artifacts": {
            "data": repo_relpath(out_pt, REPO_ROOT),
            "summary_json": repo_relpath(out_summary, REPO_ROOT),
        },
        "rho_ctrl": {
            "final_acc": scalar_stats(
                [all_data[seed]["epoch_logs"][-1]["test_acc"] for seed in seeds]
            ),
            "max_r": scalar_stats(
                [max(log["r"] for log in all_data[seed]["step_logs"]) for seed in seeds]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-18 CIFAR-10 instability detection.")
    parser.add_argument("--lrs", type=str, default="0.01,0.05,0.1,0.2",
                        help="Comma-separated learning rates to test")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log r every N batches")
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/cifar10"))
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--rho-ctrl", action="store_true",
                        help="Use rho_a controller (lr_eff = rho_a / ||g||, base_lr=inf)")
    parser.add_argument("--target-r", type=float, default=1.0,
                        help="Target r value for rho_ctrl mode (default: 1.0)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tag = f"_{args.tag}" if args.tag else ""

    # Rho-controller mode
    if args.rho_ctrl:
        target_r = args.target_r
        r_tag = f"_r{target_r:.1f}".replace(".", "p") if target_r != 1.0 else ""
        out_pt = DATA_DIR / f"resnet18_rho_ctrl{r_tag}{tag}.pt"
        out_png = PLOT_DIR / f"resnet18-rho-ctrl{r_tag}{tag}.png"
        out_pdf = PLOT_DIR / f"resnet18-rho-ctrl{r_tag}{tag}.pdf"
        out_summary = PLOT_DIR / f"summary{r_tag}{tag}.json"

        if args.replot:
            if not out_pt.exists():
                print(f"Error: {out_pt} not found")
                return
            all_data = torch.load(out_pt, weights_only=False)
            # Simple replot for rho_ctrl
            print(f"Replot not implemented for rho_ctrl mode")
            return

        print(
            f"device={device} seeds={seeds} epochs={args.epochs} "
            f"mode=rho_ctrl target_r={target_r} dataset={args.dataset}"
        )

        all_data = {}
        for seed in seeds:
            print(f"  seed {seed}")
            data = run_training(0.0, args.epochs, args.batch_size, device,
                               args.data_root, seed, args.log_every,
                               use_rho_ctrl=True, target_r=target_r,
                               dataset=args.dataset,
                               train_samples=args.train_samples,
                               test_samples=args.test_samples,
                               num_workers=args.num_workers)
            all_data[seed] = data

        torch.save(all_data, out_pt)
        write_summary(
            out_summary,
            build_rho_ctrl_summary(
                all_data,
                {
                    "seeds": seeds,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "log_every": args.log_every,
                    "target_r": target_r,
                    "dataset": args.dataset,
                    "train_samples": args.train_samples,
                    "test_samples": args.test_samples,
                    "num_workers": args.num_workers,
                },
                out_pt,
                out_summary,
            ),
        )
        print(f"\nsaved: {out_pt}")
        print(f"saved: {out_summary}")

        # Summary
        print(f"\n=== Summary (rho_ctrl target_r={target_r}, median across seeds) ===")
        max_rs = []
        final_accs = []
        for seed in seeds:
            logs = all_data[seed]['step_logs']
            max_rs.append(max(l['r'] for l in logs))
            final_accs.append(all_data[seed]['epoch_logs'][-1]['test_acc'])
        print(f"rho_ctrl(r={target_r}): max_r={np.median(max_rs):.2f}, "
              f"acc={np.median(final_accs)*100:.1f}% (IQR: {np.percentile(final_accs,25)*100:.1f}-{np.percentile(final_accs,75)*100:.1f}%)")
        return

    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]

    out_pt = DATA_DIR / f"resnet18_instability_multiseed{tag}.pt"
    out_png = PLOT_DIR / f"resnet18-instability-multiseed{tag}.png"
    out_pdf = PLOT_DIR / f"resnet18-instability-multiseed{tag}.pdf"
    out_summary = PLOT_DIR / f"summary{tag}.json"

    if args.replot:
        # Prefer momentum-corrected 5-seed data
        mom_pt = DATA_DIR / f"resnet18_instability_multiseed_momentum_5seed.pt"
        src_pt = mom_pt if mom_pt.exists() else out_pt
        if not src_pt.exists():
            print(f"Error: {src_pt} not found")
            return
        all_data = torch.load(src_pt, weights_only=False)
        for lr in all_data:
            all_data[lr] = trim_epochs(all_data[lr], 10)
        rho_variants = load_rho_variants(DATA_DIR, max_ep=10)
        make_multiseed_plot(all_data, out_png, out_pdf, rho_variants)
        print(f"saved: {out_png}")
        return

    print(
        f"device={device} lrs={lrs} seeds={seeds} epochs={args.epochs} "
        f"dataset={args.dataset}"
    )

    all_data = {lr: {} for lr in lrs}
    for lr in lrs:
        print(f"\nLR={lr}")
        for seed in seeds:
            print(f"  seed {seed}")
            data = run_training(lr, args.epochs, args.batch_size, device,
                               args.data_root, seed, args.log_every,
                               dataset=args.dataset,
                               train_samples=args.train_samples,
                               test_samples=args.test_samples,
                               num_workers=args.num_workers)
            all_data[lr][seed] = data

    # Save all data
    torch.save(all_data, out_pt)
    print(f"\nsaved: {out_pt}")

    # Multi-seed comparison plot
    make_multiseed_plot(all_data, out_png, out_pdf)
    print(f"saved: {out_png}")
    write_summary(
        out_summary,
        build_standard_summary(
            all_data,
            lrs,
            {
                "lrs": lrs,
                "seeds": seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "log_every": args.log_every,
                "dataset": args.dataset,
                "train_samples": args.train_samples,
                "test_samples": args.test_samples,
                "num_workers": args.num_workers,
            },
            out_pt,
            out_png,
            out_pdf,
            out_summary,
        ),
    )
    print(f"saved: {out_summary}")

    # Summary
    print("\n=== Summary (median across seeds) ===")
    for lr in lrs:
        max_rs = []
        r_counts = []
        final_accs = []
        for seed in seeds:
            logs = all_data[lr][seed]['step_logs']
            max_rs.append(max(l['r'] for l in logs))
            r_counts.append(sum(1 for l in logs if l['r'] > 1.0))
            final_accs.append(all_data[lr][seed]['epoch_logs'][-1]['test_acc'])
        print(f"LR={lr}: max_r={np.median(max_rs):.2f}, r>1={np.median(r_counts):.0f}, "
              f"acc={np.median(final_accs)*100:.1f}% (IQR: {np.percentile(final_accs,25)*100:.1f}-{np.percentile(final_accs,75)*100:.1f}%)")


if __name__ == "__main__":
    main()
