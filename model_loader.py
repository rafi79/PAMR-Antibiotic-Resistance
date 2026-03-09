"""
BioLinkBERT model loading and initialization.
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Tuple, List
from config import MODEL_NAME, HF_TOKEN, DEVICE, POSITIVE_WORDS, NEGATIVE_WORDS


def load_biolinkbert() -> Tuple:
    """
    Load BioLinkBERT tokenizer and masked language model.
    
    Returns:
        Tuple of (tokenizer, mlm_model, mask_token, mask_id, pos_ids, neg_ids)
    """
    print(f"\nLoading {MODEL_NAME} …")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    mlm_model.eval().to(DEVICE)
    
    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    
    n_params = sum(p.numel() for p in mlm_model.parameters()) / 1e6
    
    print(f"BioLinkBERT-base loaded ✓")
    print(f"  params     = {n_params:.0f}M")
    print(f"  vocab      = {tokenizer.vocab_size:,}")
    print(f"  mask_token = '{mask_token}'")
    print(f"  max_length = {tokenizer.model_max_length}")
    
    # Get vocabulary IDs for positive/negative words
    pos_ids = get_vocab_ids(tokenizer, POSITIVE_WORDS)
    neg_ids = get_vocab_ids(tokenizer, NEGATIVE_WORDS)
    
    print(f"\nPositive vocab tokens ({len(pos_ids)}): "
          f"{[tokenizer.convert_ids_to_tokens(i) for i in pos_ids[:6]]}")
    print(f"Negative vocab tokens ({len(neg_ids)}): "
          f"{[tokenizer.convert_ids_to_tokens(i) for i in neg_ids[:6]]}")
    
    return tokenizer, mlm_model, mask_token, mask_id, pos_ids, neg_ids


def get_vocab_ids(tokenizer, words: List[str]) -> List[int]:
    """
    Convert word list to vocabulary token IDs.
    
    Args:
        tokenizer: HuggingFace tokenizer
        words: List of words to convert
        
    Returns:
        List of token IDs
    """
    ids = []
    for w in words:
        for variant in [w, w.capitalize(), f" {w}"]:
            tid = tokenizer.convert_tokens_to_ids(variant)
            if tid not in (tokenizer.unk_token_id, None) and tid not in ids:
                ids.append(tid)
    return ids
