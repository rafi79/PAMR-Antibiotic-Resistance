"""
Full clinical report generation with formatted output.
"""
import textwrap
from typing import Dict, List
from knowledge_base import CLASS_FACTS
from report_builder import build_explanation
from citation_probes import probe


def build_report(row: dict, target_gene: str, zs_prob: float, ft_prob: float,
                true_label: int, gene_cols: List[str], class_cols: List[str],
                tokenizer, mlm_model, mask_token: str, mask_id: int, 
                device: str, ens_weight_zs: float = 0.35, 
                ens_weight_ft: float = 0.65) -> Dict:
    """
    Build complete clinical report for an isolate.
    
    Returns:
        Dictionary with report data and formatted text
    """
    gene_name = target_gene.replace("gene_", "")
    
    present_genes = [
        g.replace("gene_", "") for g in gene_cols
        if row.get(g, 0) == 1
    ]
    
    res_classes = [
        c.replace("class_", "") for c in class_cols
        if row.get(c, 0) == 1
    ]
    n_cls = len(res_classes)
    
    ens_prob = ens_weight_zs * zs_prob + ens_weight_ft * ft_prob
    predicted = int(ens_prob >= 0.5)
    conf = ens_prob if predicted == 1 else 1 - ens_prob
    correct = (predicted == true_label)
    
    if n_cls >= 10:
        tier = "EXTENSIVELY DRUG-RESISTANT (XDR)"
    elif n_cls >= 6:
        tier = "MULTI-DRUG RESISTANT (MDR)"
    elif n_cls >= 3:
        tier = "MULTI-DRUG RESISTANT (MDR)"
    else:
        tier = "LIMITED RESISTANCE"
    
    pred_str = "PRESENT" if predicted == 1 else "ABSENT"
    true_str = "PRESENT" if true_label == 1 else "ABSENT"
    result_str = "✓ CORRECT" if correct else "✗ INCORRECT"
    
    W = 72
    
    def pad(text):
        return f"║  {text:<{W-2}}║"
    
    # Run citation-aware probes
    p_sig = probe(
        f"The {gene_name} gene in E. coli is clinically {mask_token}.",
        tokenizer, mlm_model, mask_id, device, 4
    )
    p_tx = probe(
        f"E. coli with {gene_name} requires {mask_token} antibiotic treatment.",
        tokenizer, mlm_model, mask_id, device, 4
    )
    
    lines = []
    lines.append("╔" + "═"*W + "╗")
    lines.append(f"║  {'AMR CLINICAL REPORT  ·  ' + str(row.get('Isolate_ID','?')):<{W-2}}║")
    lines.append(f"║  {'Model: michiyasunaga/BioLinkBERT-base  (110M · citation-aware)':<{W-2}}║")
    lines.append("╠" + "═"*W + "╣")
    lines.append(pad(f"Organism     :  Escherichia coli"))
    lines.append(pad(f"Genome       :  {int(row.get('Genome_Length_BP',0)):,} bp  |  "
                     f"GC: {row.get('GC_Content_Percent',0):.1f}%"))
    lines.append(pad(f"AMR genes    :  {int(row.get('total_amr_genes',0))} detected"))
    lines.append(pad(f"Res. classes :  {n_cls}  →  {tier}"))
    lines.append("╠" + "═"*W + "╣")
    lines.append(pad(f"TARGET GENE  :  {gene_name}"))
    lines.append(pad(f"TRUE LABEL   :  {true_str}"))
    lines.append(pad(f"PREDICTION   :  {pred_str}  {result_str}"))
    lines.append(pad(f"Confidence   :  {conf*100:.1f}%  "
                     f"[ZS={zs_prob:.3f} | FT={ft_prob:.3f} | Ens={ens_prob:.3f}]"))
    lines.append("╠" + "═"*W + "╣")
    
    # Citation-aware probes panel
    lines.append(pad("BIOLINKBERT CITATION-AWARE PROBES:"))
    lines.append("╟" + "─"*W + "╢")
    lines.append(pad(f"  Probe: '{gene_name} in E. coli is clinically [MASK]'"))
    if p_sig:
        lines.append(pad("  → " + "  ".join(f"{t}({p*100:.1f}%)" for t, p in p_sig)))
    lines.append(pad(f"  Probe: 'E. coli with {gene_name} requires [MASK] treatment'"))
    if p_tx:
        lines.append(pad("  → " + "  ".join(f"{t}({p*100:.1f}%)" for t, p in p_tx)))
    lines.append("╠" + "═"*W + "╣")
    
    # Detected genes
    lines.append(pad("DETECTED AMR GENES:"))
    for chunk in [present_genes[i:i+4] 
                  for i in range(0, min(len(present_genes), 20), 4)]:
        lines.append(pad("  " + ", ".join(chunk)))
    lines.append("╠" + "═"*W + "╣")
    
    # Resistance classes
    lines.append(pad("RESISTANCE CLASSES:"))
    for chunk in [res_classes[i:i+5] for i in range(0, len(res_classes), 5)]:
        lines.append(pad("  " + ", ".join(chunk)))
    lines.append("╠" + "═"*W + "╣")
    
    # Clinical reasoning
    lines.append(pad("CLINICAL REASONING  (BioLinkBERT citation probes + KB):"))
    lines.append("╟" + "─"*W + "╢")
    explanation = build_explanation(
        row, target_gene, predicted, conf, gene_cols, class_cols,
        tokenizer, mlm_model, mask_token, mask_id, device
    )
    for line in textwrap.wrap(explanation, width=W-4):
        lines.append(pad(line))
    lines.append("╠" + "═"*W + "╣")
    
    # Antibiogram
    lines.append(pad("ANTIBIOGRAM INTERPRETATION:"))
    lines.append("╟" + "─"*W + "╢")
    for cls in res_classes[:9]:
        fact = CLASS_FACTS.get(cls, f"Resistance to {cls} detected.")
        tag = f"[{cls.upper()[:12]}]"
        for i, fl in enumerate(textwrap.wrap(fact, width=W-20)):
            pfx = f"  {tag:<14}" if i == 0 else f"  {' '*14}"
            lines.append(f"║{pfx}{fl:<{W-15}}║")
    lines.append("╠" + "═"*W + "╣")
    
    # Treatment
    lines.append(pad("TREATMENT RECOMMENDATION:"))
    lines.append("╟" + "─"*W + "╢")
    if "carbapenem" in res_classes and "peptide" in res_classes:
        rec = (
            "⚠ CRITICAL XDR — Immediate ID consult. Consider "
            "ceftazidime-avibactam + aztreonam, cefiderocol, or fosfomycin "
            "IV. MIC-guided combination therapy essential."
        )
    elif "carbapenem" in res_classes:
        rec = (
            "⚠ CARBAPENEM-RESISTANT — Last-resort options only. "
            "Colistin/polymyxin B, ceftazidime-avibactam (KPC), "
            "aztreonam-avibactam (MBL), temocillin (lower UTI)."
        )
    elif n_cls >= 6:
        rec = (
            "⚠ MDR — Empirical therapy will likely fail. Carbapenem "
            "for systemic infection if ESBL confirmed. Fosfomycin or "
            "nitrofurantoin for uncomplicated UTI."
        )
    else:
        rec = "Targeted therapy per susceptibility report."
    
    for line in textwrap.wrap(rec, width=W-4):
        lines.append(pad(line))
    lines.append("╠" + "═"*W + "╣")
    
    # Infection control
    lines.append(pad("INFECTION CONTROL:"))
    lines.append("╟" + "─"*W + "╢")
    mob_genes = [
        g for g in present_genes
        if any(k in g for k in ["sul", "tet", "mph", "qnr", "mcr", "dfr", 
                                "aad", "floR", "APH", "CTX", "KPC"])
    ]
    ic = (
        f"Plasmid-mediated resistance ({len(mob_genes)} genes: "
        f"{', '.join(mob_genes[:4])}). HIGH horizontal transfer risk. "
        f"Contact precautions. Screen contacts."
        if mob_genes else
        "Chromosomally-encoded resistance. Standard precautions apply."
    )
    for line in textwrap.wrap(ic, width=W-4):
        lines.append(pad(line))
    lines.append("╚" + "═"*W + "╝")
    
    return {
        "Isolate_ID": row.get("Isolate_ID", "?"),
        "True_Label": true_label,
        "ZS_Prob": round(zs_prob, 4),
        "FT_Prob": round(ft_prob, 4),
        "Ensemble_Prob": round(ens_prob, 4),
        "Predicted": predicted,
        "Confidence": round(conf, 4),
        "Correct": correct,
        "Explanation": explanation,
        "Probe_Clinical": str([t for t, _ in p_sig[:3]]),
        "Probe_Therapy": str([t for t, _ in p_tx[:3]]),
        "Report": "\n".join(lines),
    }
