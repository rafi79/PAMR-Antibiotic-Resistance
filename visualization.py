"""
Visualization dashboard for AMR results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import VIZ_COLORS, MODEL_NAME, N_GPU


def create_dashboard(results_df: pd.DataFrame, gene_data: dict, 
                     all_reports: list, best_gene_name: str,
                     output_path: str):
    """
    Create comprehensive visualization dashboard.
    
    Args:
        results_df: Results dataframe with all genes
        gene_data: Dictionary with prediction data per gene
        all_reports: List of report dictionaries
        best_gene_name: Name of best performing gene
        output_path: Path to save figure
    """
    BG = VIZ_COLORS["BG"]
    CARD = VIZ_COLORS["CARD"]
    TEXT = VIZ_COLORS["TEXT"]
    C1 = VIZ_COLORS["C1"]
    C2 = VIZ_COLORS["C2"]
    C3 = VIZ_COLORS["C3"]
    C4 = VIZ_COLORS["C4"]
    GRID = VIZ_COLORS["GRID"]
    
    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    
    fig.suptitle(
        f"AMR Clinical Intelligence — {MODEL_NAME}  (110M)\n"
        "Citation-Aware Fill-Mask Zero-Shot  ·  Fine-Tuned LOO-CV  ·  "
        "Ensemble  ·  Citation-Probe Explanations",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98,
    )
    
    def sax(ax, title):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
        ax.tick_params(colors=TEXT, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(color=GRID, lw=0.4, alpha=0.7)
    
    x = np.arange(len(results_df))
    w = 0.25
    
    # 1. AUC comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x-w, results_df["ZS_AUC"], w, color=C2, alpha=0.85,
           label="Fill-Mask ZS", edgecolor="white", lw=0.5)
    ax.bar(x, results_df["FT_AUC"], w, color=C1, alpha=0.85,
           label="Fine-Tuned FT", edgecolor="white", lw=0.5)
    ax.bar(x+w, results_df["Ens_AUC"], w, color=C4, alpha=0.85,
           label="Ensemble", edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="white", ls="--", alpha=0.3, lw=1)
    ax.axhline(0.7, color=C3, ls="--", alpha=0.5, lw=1, label="Good (0.70)")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Gene"], rotation=45, ha="right", 
                       fontsize=6, color=TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC", color=TEXT)
    ax.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT, edgecolor=GRID)
    sax(ax, "BioLinkBERT AUC: ZS vs FT vs Ensemble")
    
    # 2. Accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x-w, results_df["ZS_Acc"], w, color=C2, alpha=0.85,
           label="Fill-Mask ZS", edgecolor="white", lw=0.5)
    ax.bar(x, results_df["FT_Acc"], w, color=C1, alpha=0.85,
           label="Fine-Tuned FT", edgecolor="white", lw=0.5)
    ax.bar(x+w, results_df["Ens_Acc"], w, color=C4, alpha=0.85,
           label="Ensemble", edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="white", ls="--", alpha=0.3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Gene"], rotation=45, ha="right",
                       fontsize=6, color=TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT, edgecolor=GRID)
    sax(ax, "BioLinkBERT Accuracy: ZS vs FT vs Ensemble")
    
    # 3. Confusion matrix
    gd = gene_data[best_gene_name]
    ax = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(gd["y_true"], gd["ens_preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=["Pred:Absent", "Pred:Present"],
                yticklabels=["True:Absent", "True:Present"],
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=2, linecolor=BG, cbar=False)
    ax.set_facecolor(CARD)
    ax.set_title(f"Confusion Matrix — {best_gene_name}\n"
                 f"(BioLinkBERT Ensemble, LOO-CV)",
                 color=TEXT, fontsize=9, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    
    # 4. Confidence histogram
    ax = fig.add_subplot(gs[1, 0])
    c_ok = [r["Confidence"] for r in all_reports if r["Correct"]]
    c_err = [r["Confidence"] for r in all_reports if not r["Correct"]]
    ax.hist(c_ok, bins=12, alpha=0.85, color=C4,
            label=f"Correct (n={len(c_ok)})", edgecolor="white")
    ax.hist(c_err, bins=8, alpha=0.85, color=C2,
            label=f"Wrong (n={len(c_err)})", edgecolor="white")
    ax.axvline(0.5, color="white", ls="--", alpha=0.5)
    ax.set_xlabel("Ensemble Confidence")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT)
    sax(ax, f"Confidence Distribution — {best_gene_name}")
    
    # 5. Per-isolate strip
    ax = fig.add_subplot(gs[1, 1])
    strip_c = [C4 if r["Correct"] else C2 for r in all_reports]
    ax.bar(range(len(all_reports)), [r["Confidence"] for r in all_reports],
           color=strip_c, width=0.9, edgecolor="none")
    ax.axhline(0.5, color="white", ls="--", alpha=0.4)
    ax.set_xticks([])
    ax.set_ylabel("Confidence")
    ax.set_xlabel(f"Isolate (1–{len(all_reports)})")
    overall_acc = sum(r["Correct"] for r in all_reports) / len(all_reports)
    ax.legend(handles=[
        mpatches.Patch(color=C4, label=f"Correct ({sum(r['Correct'] for r in all_reports)})"),
        mpatches.Patch(color=C2, label=f"Wrong ({sum(not r['Correct'] for r in all_reports)})"),
    ], fontsize=8, facecolor=CARD, labelcolor=TEXT)
    sax(ax, f"Per-Isolate Strip  (Acc={overall_acc:.1%})")
    
    # 6. ZS vs FT scatter
    ax = fig.add_subplot(gs[1, 2])
    sc_c = [C4 if t == 1 else C2 for t in gd["y_true"]]
    ax.scatter(gd["zs_probs"], gd["ft_probs"],
               c=sc_c, s=80, edgecolors="white", linewidths=0.5, alpha=0.9)
    ax.axvline(0.5, color="white", ls="--", alpha=0.4)
    ax.axhline(0.5, color="white", ls="--", alpha=0.4)
    ax.set_xlabel("ZS Fill-Mask Prob.")
    ax.set_ylabel("FT Classifier Prob.")
    ax.legend(handles=[
        mpatches.Patch(color=C4, label="True: Present"),
        mpatches.Patch(color=C2, label="True: Absent"),
    ], fontsize=8, facecolor=CARD, labelcolor=TEXT)
    sax(ax, f"ZS vs FT — {best_gene_name}")
    
    # 7. Summary scorecard
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(CARD)
    ax.axis("off")
    ax.set_title("BioLinkBERT System Summary",
                 color=TEXT, fontsize=10, fontweight="bold")
    
    best_row = results_df.sort_values("Ens_AUC", ascending=False).iloc[0]
    rows = [
        ("Model", "BioLinkBERT-base"),
        ("Publisher", "michiyasunaga (ACL 2022)"),
        ("Architecture", "BERT-base (110M)"),
        ("Pre-training", "PubMed + citation links"),
        ("Max length", "512 tokens"),
        ("GPU", f"GPU × {N_GPU}"),
        ("", ""),
        ("Method A", "Fill-mask zero-shot"),
        ("Method B", "Fine-tuned LOO-CV"),
        ("Method C", "Ensemble (0.35+0.65)"),
        ("Explanation", "Citation probes + KB"),
        ("", ""),
        ("Targets", f"{len(results_df)} variable genes"),
        ("Best gene", best_row["Gene"]),
        ("Best Ens AUC", f"{best_row['Ens_AUC']:.3f}"),
        ("Best FT AUC", f"{best_row['FT_AUC']:.3f}"),
        ("Report Acc.", f"{overall_acc:.1%}"),
    ]
    
    for j, (k, v) in enumerate(rows):
        yp = 9.8 - j * 0.54
        if k:
            ax.text(0.3, yp, k+":", color="#607a99", fontsize=7.5, va="top")
            ax.text(5.2, yp, v, color=TEXT, fontsize=7.5,
                    va="top", fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved → {output_path}")
