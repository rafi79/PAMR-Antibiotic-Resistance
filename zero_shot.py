"""
Zero-shot fill-mask prediction using BioLinkBERT.
"""
import torch
from typing import Tuple, List
from knowledge_base import GENE_FACTS


def build_fillmask_prompt(row: dict, target_gene: str, 
                          gene_cols: List[str], class_cols: List[str],
                          mask_token: str) -> str:
    """
    Build fill-mask prompt with [MASK] in target position.
    
    Args:
        row: Data row dictionary
        target_gene: Target gene column name
        gene_cols: List of all gene columns
        class_cols: List of all class columns
        mask_token: Mask token string
        
    Returns:
        Prompt string with [MASK]
    """
    gene_name = target_gene.replace("gene_", "")
    
    other_genes = [
        g.replace("gene_", "") for g in gene_cols
        if g != target_gene and row.get(g, 0) == 1
    ]
    
    res_classes = [
        c.replace("class_", "") for c in class_cols
        if row.get(c, 0) == 1
    ]
    
    # Get mechanism from knowledge base
    facts = [GENE_FACTS[g] for g in other_genes[:3] if g in GENE_FACTS]
    mech = facts[0][:90] if facts else "multidrug resistance mechanisms"
    
    return (
        f"This Escherichia coli clinical isolate carries resistance genes "
        f"{', '.join(other_genes[:10]) or 'none'} conferring resistance to "
        f"{', '.join(res_classes[:8]) or 'no tested classes'}. "
        f"Key mechanism: {mech}. "
        f"The {gene_name} antimicrobial resistance gene is {mask_token} "
        f"in this isolate."
    )


def zero_shot_score(prompt: str, tokenizer, mlm_model, 
                    mask_id: int, pos_ids: List[int], 
                    neg_ids: List[int], device: str) -> Tuple[float, int]:
    """
    Score [MASK] position using positive vs negative vocabulary.
    
    Args:
        prompt: Prompt with [MASK] token
        tokenizer: HuggingFace tokenizer
        mlm_model: Masked language model
        mask_id: Mask token ID
        pos_ids: Positive vocabulary IDs
        neg_ids: Negative vocabulary IDs
        device: Device string
        
    Returns:
        Tuple of (score, prediction)
    """
    enc = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True, 
        max_length=512,
    ).to(device)
    
    mask_pos = (enc["input_ids"] == mask_id).nonzero(as_tuple=True)[1]
    if len(mask_pos) == 0:
        return 0.5, 0
    
    with torch.no_grad():
        logits = mlm_model(**enc).logits
        probs = torch.softmax(logits[0, mask_pos[0].item(), :], dim=-1).cpu()
    
    p_pos = sum(probs[i].item() for i in pos_ids if i < len(probs))
    p_neg = sum(probs[i].item() for i in neg_ids if i < len(probs))
    
    total = p_pos + p_neg + 1e-9
    score = p_pos / total
    
    return float(score), int(score >= 0.5)
