"""
Main execution script for AMR BioLinkBERT analysis.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

from config import DATA_PATH, OUTPUT_DIR, ZERO_SHOT_WEIGHT, FINE_TUNED_WEIGHT, DEVICE
from data_loader import load_and_preprocess_data
from model_loader import load_biolinkbert
from zero_shot import build_fillmask_prompt, zero_shot_score
from fine_tuning import run_loo_biolinkbert
from clinical_report import build_report
from visualization import create_dashboard


def main():
    """Main execution function."""
    print("="*65)
    print("AMR CLINICAL INTELLIGENCE — BioLinkBERT Edition (V10)")
    print("michiyasunaga/BioLinkBERT-base")
    print("="*65)
    
    # Load data
    print("\n[1/6] Loading and preprocessing data...")
    df, gene_cols, class_cols, variable_genes = load_and_preprocess_data(DATA_PATH)
    
    # Load model
    print("\n[2/6] Loading BioLinkBERT model...")
    tokenizer, mlm_model, mask_token, mask_id, pos_ids, neg_ids = load_biolinkbert()
    
    # Evaluate all genes
    print("\n[3/6] Evaluating all variable genes...")
    print("="*65)
    print("BIOLINKBERT AMR EVALUATION — ALL VARIABLE GENES")
    print("="*65)
    
    all_results = []
    gene_data = {}
    
    for gene, y_true in variable_genes.items():
        gene_name = gene.replace("gene_", "")
        print(f"\n{'─'*65}")
        print(f"GENE: {gene_name}   pos={y_true.sum()}  neg={50-y_true.sum()}")
        print(f"{'─'*65}")
        
        # A: Zero-shot
        print("  Zero-shot fill-mask …")
        zs_probs, zs_preds = [], []
        for _, row in df.iterrows():
            prompt = build_fillmask_prompt(
                row.to_dict(), gene, gene_cols, class_cols, mask_token
            )
            p, pred = zero_shot_score(
                prompt, tokenizer, mlm_model, mask_id, pos_ids, neg_ids, DEVICE
            )
            zs_probs.append(p)
            zs_preds.append(pred)
        
        zs_probs = np.array(zs_probs)
        zs_preds = np.array(zs_preds)
        zs_acc = accuracy_score(y_true, zs_preds)
        try:
            zs_auc = roc_auc_score(y_true, zs_probs)
        except:
            zs_auc = float("nan")
        
        print(f"  Zero-Shot  → Acc={zs_acc:.3f}  AUC={zs_auc:.3f}  "
              f"pred_pos={zs_preds.sum()}")
        
        # B: Fine-tuned LOO
        print("  Fine-tuned LOO-CV (50 folds) …")
        ft_preds, ft_probs = run_loo_biolinkbert(
            df, gene, y_true, gene_cols, class_cols
        )
        ft_acc = accuracy_score(y_true, ft_preds)
        try:
            ft_auc = roc_auc_score(y_true, ft_probs)
        except:
            ft_auc = float("nan")
        
        print(f"  Fine-Tuned → Acc={ft_acc:.3f}  AUC={ft_auc:.3f}  "
              f"pred_pos={ft_preds.sum()}")
        print(classification_report(y_true, ft_preds, zero_division=0))
        
        # C: Ensemble
        ens_probs = ZERO_SHOT_WEIGHT * zs_probs + FINE_TUNED_WEIGHT * ft_probs
        ens_preds = (ens_probs >= 0.5).astype(int)
        ens_acc = accuracy_score(y_true, ens_preds)
        try:
            ens_auc = roc_auc_score(y_true, ens_probs)
        except:
            ens_auc = float("nan")
        
        print(f"  Ensemble   → Acc={ens_acc:.3f}  AUC={ens_auc:.3f}")
        
        all_results.append({
            "Gene": gene_name,
            "Pos": int(y_true.sum()),
            "ZS_Acc": round(zs_acc, 4),
            "ZS_AUC": round(zs_auc, 4),
            "FT_Acc": round(ft_acc, 4),
            "FT_AUC": round(ft_auc, 4),
            "Ens_Acc": round(ens_acc, 4),
            "Ens_AUC": round(ens_auc, 4),
        })
        
        gene_data[gene_name] = {
            "y_true": y_true,
            "zs_probs": zs_probs,
            "zs_preds": zs_preds,
            "ft_probs": ft_probs,
            "ft_preds": ft_preds,
            "ens_probs": ens_probs,
            "ens_preds": ens_preds,
        }
    
    results_df = pd.DataFrame(all_results)
    print("\n\n=== FINAL RESULTS ===")
    print(results_df.to_string(index=False))
    
    # Generate clinical reports
    print("\n[4/6] Generating clinical reports...")
    print("="*65)
    print("GENERATING CLINICAL REPORTS — ALL 50 ISOLATES")
    print("="*65)
    
    best_gene_name = results_df.sort_values("Ens_AUC", ascending=False).iloc[0]["Gene"]
    best_gene = f"gene_{best_gene_name}"
    gd = gene_data[best_gene_name]
    
    all_reports = []
    for i, (_, row) in enumerate(df.iterrows()):
        true_label = int(gd["y_true"][i])
        print(f"\n[{i+1:02d}/50] {row.get('Isolate_ID','?')}  "
              f"True={'P' if true_label else 'A'}  "
              f"ZS={gd['zs_probs'][i]:.3f}  FT={gd['ft_probs'][i]:.3f}")
        
        rep = build_report(
            row.to_dict(), best_gene,
            float(gd["zs_probs"][i]),
            float(gd["ft_probs"][i]),
            true_label,
            gene_cols, class_cols,
            tokenizer, mlm_model, mask_token, mask_id, DEVICE,
            ZERO_SHOT_WEIGHT, FINE_TUNED_WEIGHT
        )
        print(rep["Report"])
        all_reports.append(rep)
    
    reports_df = pd.DataFrame(all_reports)
    overall_acc = reports_df["Correct"].mean()
    print(f"\n\nFinal accuracy ({best_gene_name}): {overall_acc:.1%}  "
          f"({int(reports_df['Correct'].sum())}/50)")
    
    # Create visualization
    print("\n[5/6] Creating visualization dashboard...")
    create_dashboard(
        results_df, gene_data, all_reports, best_gene_name,
        f"{OUTPUT_DIR}/amr_biolinkbert.png"
    )
    
    # Save outputs
    print("\n[6/6] Saving outputs...")
    results_df.to_csv(f"{OUTPUT_DIR}/amr_biolinkbert_results.csv", index=False)
    
    with open(f"{OUTPUT_DIR}/amr_biolinkbert_reports.txt", "w", 
              encoding="utf-8") as f:
        for r in all_reports:
            f.write(r["Report"] + "\n\n")
            f.write(f"  Citation probe (clinical) : {r['Probe_Clinical']}\n")
            f.write(f"  Citation probe (therapy)  : {r['Probe_Therapy']}\n\n")
            f.write("~"*74 + "\n\n")
    
    pd.DataFrame([
        {k: v for k, v in r.items() if k != "Report"}
        for r in all_reports
    ]).to_csv(f"{OUTPUT_DIR}/amr_biolinkbert_summary.csv", index=False)
    
    print("\n✓  amr_biolinkbert_results.csv  — AUC/Acc all genes × all methods")
    print("✓  amr_biolinkbert_reports.txt  — Full clinical reports + probes")
    print("✓  amr_biolinkbert_summary.csv  — Per-isolate predictions")
    print("✓  amr_biolinkbert.png          — 8-panel dashboard")
    print(f"\nFinal ensemble accuracy ({best_gene_name}): {overall_acc:.1%}")


if __name__ == "__main__":
    main()
