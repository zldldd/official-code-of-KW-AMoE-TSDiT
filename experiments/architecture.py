"""Draw architecture schematics from the implemented route-A computation path."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, width, height, text, color="#e8f1fb"):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015", fc=color, ec="#23405e", lw=1.3)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)
    return (x, y, width, height)


def arrow(ax, start, end, text=None, color="#23405e", connectionstyle="arc3,rad=0.0"):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.25, color=color, connectionstyle=connectionstyle)
    ax.add_patch(patch)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, text, fontsize=8, ha="center", color=color)


def canvas(title):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=15)
    return fig, ax


def overview(output):
    fig, ax = canvas("KW-AMoE-TSDiT: implemented route-A data flow")
    box(ax, (0.02, 0.63), 0.18, 0.15, "$z_\\tau$, DDIM step $\\tau$\nview / latent-time embeddings")
    box(ax, (0.29, 0.63), 0.2, 0.15, "DiT blocks\nGlobal MHSA + adaLN")
    box(ax, (0.58, 0.63), 0.19, 0.15, "KW-AMoE\nshared $s_t,M_t,w_t,K_t$")
    box(ax, (0.81, 0.63), 0.17, 0.15, "Velocity\n$\\hat v_{final}$")
    arrow(ax, (0.2, 0.705), (0.29, 0.705))
    arrow(ax, (0.49, 0.705), (0.58, 0.705), "frame tokens")
    arrow(ax, (0.77, 0.705), (0.81, 0.705), "$K_t=1$ or $K_e$")
    box(ax, (0.15, 0.19), 0.22, 0.17, "TS-Mem\nmask-weighted writes")
    box(ax, (0.45, 0.19), 0.23, 0.17, "Readout $R_t$\n$C_t=\\beta(1-M_t)R_t$")
    box(ax, (0.77, 0.19), 0.2, 0.17, "Three feedback paths\nrouter / adaLN / $W_{mem}C$")
    arrow(ax, (0.68, 0.63), (0.34, 0.36), "$w_t,U_t$")
    arrow(ax, (0.37, 0.275), (0.45, 0.275), "cross-attention")
    arrow(ax, (0.68, 0.275), (0.77, 0.275))
    arrow(ax, (0.85, 0.36), (0.87, 0.63), "residual", connectionstyle="arc3,rad=-0.22")
    arrow(ax, (0.58, 0.36), (0.42, 0.63), connectionstyle="arc3,rad=0.2")
    arrow(ax, (0.56, 0.36), (0.66, 0.63), connectionstyle="arc3,rad=-0.2")
    ax.text(0.41, 0.52, "adaLN", fontsize=8, ha="center")
    ax.text(0.63, 0.52, "$r_b=\\sum_t w_tR_t$", fontsize=8, ha="center")
    ax.text(0.5, 0.055, "Memory is reset once per training sample or full generated sequence; non-keyframes have zero forward write weight.", ha="center", fontsize=10)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def kwamoe(output):
    fig, ax = canvas("Unified keyframe state and actual expert budgets")
    box(ax, (0.02, 0.62), 0.17, 0.17, "$U_t$\ntemporal MHSA\n+ frame pooling")
    box(ax, (0.26, 0.62), 0.16, 0.17, "$s_t=MLP_{sal}(U_t)$\none score")
    box(ax, (0.49, 0.69), 0.19, 0.14, "$p_t=softmax(s_t)$\n$M_t=TopK(s_t)$")
    box(ax, (0.49, 0.42), 0.19, 0.15, "$w_t=M_t^{ST}p_t/\\sum_jM_j^{ST}p_j$")
    box(ax, (0.75, 0.62), 0.2, 0.17, "$K_t=1+(K_e-1)M_t$\nkey: $K_e$; non-key: 1")
    arrow(ax, (0.19, 0.705), (0.26, 0.705))
    arrow(ax, (0.42, 0.705), (0.49, 0.76), "same $s_t$")
    arrow(ax, (0.585, 0.69), (0.585, 0.57), "mask + probability")
    arrow(ax, (0.68, 0.495), (0.75, 0.68), "$M_t=0\\Rightarrow w_t=0$")
    box(ax, (0.22, 0.14), 0.23, 0.15, "$h^{global}=\\sum_t w_t\\bar h_t$\nframe + memory context")
    box(ax, (0.54, 0.14), 0.27, 0.15, "TopK($g_t,K_t$) per frame\nactual expert calls + FLOPs")
    arrow(ax, (0.49, 0.495), (0.35, 0.29), "$w_t$")
    arrow(ax, (0.45, 0.215), (0.54, 0.215), "router")
    arrow(ax, (0.76, 0.62), (0.71, 0.29), "budget")
    ax.text(0.5, 0.04, "$\\sum_t w_t=1$; support consistency, not a guarantee that every keyframe exceeds $1/T$.", ha="center", fontsize=10)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def tsmem(output):
    fig, ax = canvas("TS-Mem: mask-weighted writes and three feedback paths")
    box(ax, (0.03, 0.61), 0.21, 0.17, "$c_{w,t}=W_{write}U_t$\n$a_t=\\alpha\\,clip(Tw_t,0,1)$")
    box(ax, (0.33, 0.61), 0.25, 0.17, "$M_t^{new}=(1-a_t)M_t^{old}+a_tc_{w,t}$\nreset once per full sequence")
    box(ax, (0.68, 0.61), 0.27, 0.17, "$R_t=W_{read}(CrossAttn(U_t,M,M)+M)$")
    arrow(ax, (0.24, 0.695), (0.33, 0.695), "$w_t,U_t$")
    arrow(ax, (0.58, 0.695), (0.68, 0.695))
    box(ax, (0.07, 0.2), 0.2, 0.16, "$r_b=\\sum_t w_tR_t$\nrouter context")
    box(ax, (0.4, 0.2), 0.22, 0.16, "$C_t=\\beta(1-M_t)R_t$\nadaLN conditioning")
    box(ax, (0.74, 0.2), 0.21, 0.16, "$\\hat v_{final}=\\hat v_{DiT}+W_{mem}C$\nnon-key residual")
    arrow(ax, (0.77, 0.61), (0.18, 0.36), "weighted readout", connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.79, 0.61), (0.51, 0.36), "memory context", connectionstyle="arc3,rad=0.08")
    arrow(ax, (0.82, 0.61), (0.84, 0.36), "non-key only")
    ax.text(0.5, 0.055, "The gate $Tw_t$ is not multiplied into the write candidate a second time. No real future latent is fed back.", ha="center", fontsize=10)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    output = Path("output/mini/report/architecture")
    output.mkdir(parents=True, exist_ok=True)
    overview(output / "fig2_overview.png")
    kwamoe(output / "fig3_kwamoe.png")
    tsmem(output / "fig4_tsmem.png")
    print(f"Wrote route-A diagrams to {output}")


if __name__ == "__main__":
    main()
