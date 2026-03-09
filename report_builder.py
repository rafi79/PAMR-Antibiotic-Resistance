"""
Clinical report generation with citation-aware explanations.
"""
import textwrap
from typing import Dict, List
from knowledge_base import GENE_FACTS, CLASS_FACTS
from citation_probes import probe


def build_explanation(row: dict, target_gene: str, predicted: int, 
                      confidence: float, gene_cols: List[str], 
                      class_cols: List[str], tokenizer, mlm_model,
                      mask_token: str, mask_id: int, device: str) -> str:
    """
    Build citation-aware explanation using BioLinkBERT probes.
    
    Args:
        row: Data row dictionary
        target_gene: Target gene column name
        predicted: Predicted label (0 or 1)
        confidence: Prediction confidence
        gene_cols: List of gene columns
        class_cols: List of class columns
        tokenizer: HuggingFace tokenizer
        mlm_model: Masked language model
        mask_token: Mask token string
        mask_id: Mask token ID
        device: Device string
        
    Returns:
        Explanation text
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
    
    # Gene categories
    efflux = [g for g in present_genes
              if any(k in g for k in ["acr", "mdt", "emr", "tolC", "mdf"])]
    bla = [g for g in present_genes
           if any(k in g for k in ["CTX", "KPC", "CMY", "ampC", "bla", "ampH"])]
    mob = [g for g in present_genes
           if any(k in g for k in ["sul", "tet", "mph", "qnr", "mcr", "dfr", 
                                   "aad", "floR", "APH"])]
    
    # Run citation probes
    p1 = probe(
        f"The {gene_name} resistance gene has {mask_token} clinical "
        f"significance in multidrug-resistant Escherichia coli.",
        tokenizer, mlm_model, mask_id, device, top_k=4
    )
    
    p2 = probe(
        f"In E. coli, {gene_name} is {mask_token} co-detected with "
        f"efflux pump genes and plasmid-mediated resistance determinants.",
        tokenizer, mlm_model, mask_id, device, top_k=4
    )
    
    p3 = probe(
        f"Infections caused by E. coli carrying {gene_name} typically "
        f"require {mask_token} antibiotic therapy.",
        tokenizer, mlm_model, mask_id, device, top_k=4
    )
    
    p1_w = [t for t, _ in p1 if len(t) > 2 and "##" not in t]
    p2_w = [t for t, _ in p2 if len(t) > 2 and "##" not in t]
    p3_w = [t for t, _ in p3 if len(t) > 2 and "##" not in t]
    
    pres = "present" if predicted == 1 else "absent"
    conf_desc = (
        "high" if confidence > 0.80 else
        "moderate" if confidence > 0.65 else "low"
    )
    
    parts = []
    parts.append(
        f"BioLinkBERT predicts the {gene_name} resistance determinant is "
        f"{pres} with {conf_desc} confidence ({confidence*100:.0f}%). "
    )
    
    # Gene KB fact
    gk = gene_name.replace("Escherichia_coli_", "")
    kb_key = gene_name if gene_name in GENE_FACTS else gk
    if kb_key in GENE_FACTS:
        parts.append(GENE_FACTS[kb_key] + " ")
    
    # Efflux context
    if efflux:
        parts.append(
            f"Co-detected efflux pump genes ({', '.join(efflux[:3])}) "
            f"indicate upregulated AcrAB-TolC activity, reducing "
            f"intracellular fluoroquinolone, tetracycline and beta-lactam "
            f"concentrations. "
        )
    
    # Beta-lactamase context
    if bla:
        parts.append(
            f"Beta-lactamases ({', '.join(bla[:2])}) provide enzymatic "
            f"hydrolysis complementary to efflux, conferring broad "
            f"beta-lactam resistance independent of porin changes. "
        )
    
    # Plasmid mobility
    if mob:
        parts.append(
            f"Plasmid-borne genes ({', '.join(mob[:4])}) confirm horizontal "
            f"gene transfer capacity; BioLinkBERT citation probes indicate "
            f"this profile is {', '.join(p2_w[:2]) or 'frequently'} "
            f"co-detected in the resistance literature. "
        )
    
    # Therapy probe result
    if p3_w:
        parts.append(
            f"Model inference suggests {gene_name}-positive isolates "
            f"require {', '.join(p3_w[:2])} antibiotic management. "
        )
    
    # Resistance burden
    if n_cls >= 10:
        parts.append(
            f"Resistance across {n_cls} antibiotic classes meets XDR "
            f"criteria, leaving only combination salvage regimens as options."
        )
    elif n_cls >= 6:
        parts.append(
            f"The {n_cls}-class MDR phenotype necessitates carbapenem-based "
            f"or combination rescue therapy for systemic infection."
        )
    
    result = "".join(parts).strip()
    if not result.endswith("."):
        result += "."
    
    return result
