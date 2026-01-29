#!/usr/bin/env python3
import argparse
import os
from copy import deepcopy

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from src.data import generate_data, get_promt, random_sample_data
from src.transformer_general import Transformer_F, in_context_loss, clip_and_step
from src.utils import get_device, parse_seed_list

matplotlib.use("Agg")

def build_optimizer(alg: str, parameters, lr: float):
    if alg == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0)
    if alg == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.9), weight_decay=0.0)
    raise ValueError("alg must be 'sgd' or 'adam'")


def set_trainable_params(model: Transformer_F, method: str) -> str:
    base = method.split("_", 1)[0]
    suffix = method.split("_", 1)[1] if "_" in method else ""

    model.v_matrix.requires_grad_(True)
    model.q_matrix.requires_grad_(True)
    model.k_matrix.requires_grad_(True)

    if suffix == "":
        return base
    if suffix == "v":
        model.q_matrix.requires_grad_(False)
        model.k_matrix.requires_grad_(False)
        model.v_matrix.requires_grad_(True)
        return base
    if suffix == "kq":
        model.v_matrix.requires_grad_(False)
        model.q_matrix.requires_grad_(True)
        model.k_matrix.requires_grad_(True)
        return base
    if suffix == "k":
        model.v_matrix.requires_grad_(False)
        model.q_matrix.requires_grad_(False)
        model.k_matrix.requires_grad_(True)
        return base
    if suffix == "q":
        model.v_matrix.requires_grad_(False)
        model.k_matrix.requires_grad_(False)
        model.q_matrix.requires_grad_(True)
        return base
    raise ValueError(f"Unknown fine-tune suffix: {suffix!r} (method={method!r})")


def make_run_dir(args) -> str:
    run_name = (
        f"h{args.n_head}_l{args.n_layer}_d{args.d}_"
        f"m{args.N_pt}_n{args.N_ft}_{args.alg}_"
        f"lrpt{args.lr_pt}_lrft{args.lr_ft}_"
        f"itpt{args.max_iters_pt}_itft{args.max_iters_ft}"
    )
    if not args.save_dir:
        return os.getcwd()
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_fig(fig, out_dir: str, stem: str, dpi: int = 300):
    # `bbox_inches="tight"` can still clip tick labels in edge cases (esp. with log ticks);
    # add a small padding to make the output robust.
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(
        os.path.join(out_dir, f"{stem}.png"), bbox_inches="tight", pad_inches=0.1, dpi=dpi
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--alg", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr-pt", type=float, required=True)
    parser.add_argument("--lr-ft", type=float, required=True)
    parser.add_argument("--max-iters-pt", type=int, required=True)
    parser.add_argument("--max-iters-ft", type=int, required=True)
    parser.add_argument("--B-pt", type=int, required=True)
    parser.add_argument("--B-ft", type=int, required=True)
    parser.add_argument("--N-pt", type=int, required=True)
    parser.add_argument("--N-ft", type=int, required=True)
    parser.add_argument("--dataset-size-ft", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--x-std", type=float, required=True)
    parser.add_argument("--noise-std", type=float, required=True)
    parser.add_argument("--B-eval", type=int, required=True)
    parser.add_argument("--n-shots-max", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seeds", type=str, default="0:5")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams.update({"font.size": 18})
    legend_fontsize = 14

    device = get_device(args.device)
    test_seeds = parse_seed_list(args.test_seeds)

    out_dir = make_run_dir(args)

    ft_methods = ["zs", "both", "zs_q", "both_q", "zs_k", "both_k", "zs_v", "both_v"]
    labels = {
        "zs": "ZS (all)",
        "both": "ZS+FS (all)",
        "zs_q": "ZS (Q)",
        "both_q": "ZS+FS (Q)",
        "zs_k": "ZS (K)",
        "both_k": "ZS+FS (K)",
        "zs_v": "ZS (V)",
        "both_v": "ZS+FS (V)",
    }

    # ---------- pretrain ----------
    torch.manual_seed(args.train_seed)
    model_pretrained = Transformer_F(args.n_layer, args.n_head, args.d).to(device)
    optimizer = build_optimizer(args.alg, model_pretrained.parameters(), args.lr_pt)

    pretrain_losses: list[float] = []
    for _ in range(args.max_iters_pt):
        z, y_test = generate_data(
            args.B_pt,
            args.N_pt + 1,
            args.d,
            x_std=args.x_std,
            noise_std=args.noise_std,
            use_prompt=True,
            device=device,
        )
        loss = in_context_loss(model_pretrained, z, y_test)
        pretrain_losses.append(float(loss.item()))
        loss.backward()
        clip_and_step(model_pretrained, optimizer, True)
        optimizer.zero_grad(set_to_none=True)

    pretrained_params = model_pretrained.allparam.detach().clone()

    # ---------- finetune ----------
    theta0 = torch.randn(args.d, device=device)
    theta0 = theta0 / theta0.norm().clamp_min(1e-12)

    x_dataset_ft, y_dataset_ft = generate_data(
        1,
        args.dataset_size_ft,
        args.d,
        theta=theta0,
        x_std=args.x_std,
        noise_std=args.noise_std,
        device=device,
    )
    x_dataset_ft, y_dataset_ft = x_dataset_ft.squeeze(0), y_dataset_ft.squeeze(0)

    finetune_losses: dict[str, list[float]] = {}
    finetuned_params: dict[str, torch.Tensor] = {}

    for method in ft_methods:
        model = deepcopy(model_pretrained)
        method_loss = set_trainable_params(model, method)
        optimizer = build_optimizer(args.alg, model.parameters(), args.lr_ft)

        losses: list[float] = []
        for t in range(args.max_iters_ft):
            x, y = random_sample_data(
                x_dataset_ft, y_dataset_ft, args.B_ft, args.N_ft + 1, device=device
            )
            z_context, y_test = get_promt(x, y)
            z = z_context[:, [-1], :]

            zs_loss = in_context_loss(model, z, y_test)
            fs_loss, temp_loss_weight = zs_loss, 0.0

            if method_loss == "zs":
                loss = zs_loss
            elif method_loss == "both":
                fs_loss = in_context_loss(model, z_context, y_test)
                temp_loss_weight = (
                    (args.max_iters_ft - 1 - t) / (args.max_iters_ft - 1)
                    if args.max_iters_ft > 1
                    else 0.0
                )
                loss = zs_loss + temp_loss_weight * fs_loss
            else:
                raise ValueError("ft method must start with 'zs' or 'both'")

            losses.append(float(loss.item()))
            loss.backward()
            clip_and_step(model, optimizer, True)
            optimizer.zero_grad(set_to_none=True)

        finetune_losses[method] = losses
        finetuned_params[method] = model.allparam.detach().clone()

    # ---------- figure 1: train losses ----------
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(pretrain_losses)
    axs[0].set_title("Pretrain train loss")
    axs[0].set_xlabel("iter")
    axs[0].set_ylabel("loss")
    axs[0].set_yscale("log")
    axs[0].grid(alpha=0.3)

    for method in ft_methods:
        axs[1].plot(finetune_losses[method], label=labels.get(method, method))
    axs[1].set_title("Finetune train loss (all methods)")
    axs[1].set_xlabel("iter")
    axs[1].set_ylabel("loss")
    axs[1].set_yscale("log")
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=legend_fontsize)

    plt.tight_layout()
    save_fig(fig, out_dir, "train_losses")
    if args.show:
        plt.show()
    plt.close(fig)

    # ---------- eval ----------
    N_list = list(range(0, args.n_shots_max + 1))
    model_eval = Transformer_F(args.n_layer, args.n_head, args.d).to(device)

    results: dict[str, dict[str, np.ndarray]] = {}
    for method in ["pretrained"] + ft_methods:
        params = pretrained_params if method == "pretrained" else finetuned_params[method]
        with torch.no_grad():
            model_eval.allparam = params

        pos_seeds, neg_seeds = [], []
        for seed in test_seeds:
            torch.manual_seed(int(seed))
            pos_vals, neg_vals = [], []
            for N in N_list:
                z, y = generate_data(
                    args.B_eval,
                    N + 1,
                    args.d,
                    theta=theta0,
                    x_std=args.x_std,
                    noise_std=args.noise_std,
                    use_prompt=True,
                    device=device,
                )
                with torch.no_grad():
                    pos_vals.append(float(in_context_loss(model_eval, z, y).item()))

                z, y = generate_data(
                    args.B_eval,
                    N + 1,
                    args.d,
                    theta=-theta0,
                    x_std=args.x_std,
                    noise_std=args.noise_std,
                    use_prompt=True,
                    device=device,
                )
                with torch.no_grad():
                    neg_vals.append(float(in_context_loss(model_eval, z, y).item()))

            pos_seeds.append(torch.tensor(pos_vals))
            neg_seeds.append(torch.tensor(neg_vals))

        results[method] = {
            "learned_pos": torch.stack(pos_seeds).mean(dim=0).cpu().numpy(),
            "learned_neg": torch.stack(neg_seeds).mean(dim=0).cpu().numpy(),
        }

    # ---------- figure 2: comparison (log y-axis) ----------
    metrics = [("learned_pos", r"$\theta_0$"), ("learned_neg", r"$-\theta_0$")]
    colors = {"full": "#4D96FF", "k": "#6BCB77", "q": "#FFD93D", "v": "#FF6B6B"}

    def style(m: str):
        c = (
            colors["k"]
            if m.endswith("_k")
            else colors["q"]
            if m.endswith("_q")
            else colors["v"]
            if m.endswith("_v")
            else colors["full"]
        )
        return c, m.startswith("both")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax_l, ax_r = ax
    handles: list[Line2D] = []

    # When values are very large but tightly clustered, a sparse LogLocator (e.g. only 1/3/5)
    # can produce no ticks within the visible range, making the left y-axis look "blank".
    # Add a bit of multiplicative padding so there is always at least one major tick.
    all_y_vals: list[float] = []
    for metric, _t in metrics:
        for m in ft_methods:
            all_y_vals.extend(results[m][metric].tolist())
    y_arr = np.asarray(all_y_vals, dtype=float)
    y_arr = y_arr[np.isfinite(y_arr) & (y_arr > 0)]
    if y_arr.size:
        y_min = float(y_arr.min())
        y_max = float(y_arr.max())
        if y_min == y_max:
            y_min, y_max = y_min / 10.0, y_max * 10.0
        else:
            pad = 1.6
            y_min, y_max = y_min / pad, y_max * pad

    for j, ((metric, t), a) in enumerate(zip(metrics, ax)):
        for m in ft_methods:
            c, filled = style(m)
            a.scatter(
                N_list,
                results[m][metric],
                facecolors=c if filled else "none",
                edgecolors=c,
                linewidths=1.5,
                s=30,
            )

            if j == 0:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        markerfacecolor=c if filled else "none",
                        markeredgecolor=c,
                        linestyle="None",
                        markersize=7,
                    )
                )

        a.set_yscale("log")
        a.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 3.0, 5.0)))
        a.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        a.yaxis.set_minor_formatter(mticker.NullFormatter())
        a.grid(alpha=0.3)
        a.set_xticks([0, 10, 20])
        a.set_title(rf"evaluate on the task {t}", fontsize=legend_fontsize, pad=6)

    ax_l.set_ylabel("Test error")
    if y_arr.size:
        ax_l.set_ylim(y_min, y_max)
    ax_l.tick_params(axis="y", which="both", labelleft=True)
    ax_r.tick_params(axis="y", which="both", labelleft=False, labelright=False)
    fig.supxlabel("Number of shots (n)", fontsize=18)
    fig.legend(
        handles,
        [labels[m] for m in ft_methods],
        ncol=4,
        fontsize=legend_fontsize,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08)
    )
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.14, right=0.98, wspace=0.12)

    save_fig(fig, out_dir, "linear_tf_thetas_qkv")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
