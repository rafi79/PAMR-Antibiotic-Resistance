"""
Citation-aware probing using BioLinkBERT's fill-mask capabilities.
"""
import torch
from typing import List, Tuple


def probe(statement: str, tokenizer, mlm_model, mask_id: int, 
          device: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Probe [MASK] position and return top-k predictions.
    
    Args:
        statement: Statement with [MASK] token
        tokenizer: HuggingFace tokenizer
        mlm_model: Masked language model
        mask_id: Mask token ID
        device: Device string
        top_k: Number of top predictions to return
        
    Returns:
        List of (token, probability) tuples
    """
    enc = tokenizer(
        statement, 
        return_tensors="pt",
        truncation=True, 
        max_length=512,
    ).to(device)
    
    mask_positions = (enc["input_ids"] == mask_id).nonzero(as_tuple=True)[1]
    if len(mask_positions) == 0:
        return []
    
    with torch.no_grad():
        logits = mlm_model(**enc).logits
        probs = torch.softmax(logits[0, mask_positions[0].item(), :], dim=-1)
        top = torch.topk(probs, top_k)
    
    return [
        (tokenizer.convert_ids_to_tokens(idx.item()), prob.item())
        for idx, prob in zip(top.indices, top.values)
    ]
