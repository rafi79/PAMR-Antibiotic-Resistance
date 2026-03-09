"""
Fine-tuned sequence classification with LOO-CV.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from typing import Tuple
from knowledge_base import GENE_FACTS
from config import (
    MODEL_NAME, HF_TOKEN, DEVICE, N_GPU,
    EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_LENGTH
)


class AMRDataset(Dataset):
    """Dataset for AMR classification."""
    
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def build_clf_prompt(row: dict, target_gene: str, 
                     gene_cols: list, class_cols: list) -> str:
    """
    Build classification prompt (no [MASK]).
    
    Args:
        row: Data row dictionary
        target_gene: Target gene column name
        gene_cols: List of all gene columns
        class_cols: List of all class columns
        
    Returns:
        Classification prompt string
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
    
    facts = [GENE_FACTS[g] for g in other_genes[:4] if g in GENE_FACTS]
    mech = "; ".join(f[:70] for f in facts[:2]) if facts else "multidrug resistance"
    
    return (
        f"Clinical microbiology: Escherichia coli. "
        f"Genome {int(row.get('Genome_Length_BP', 0)):,} bp, "
        f"GC {row.get('GC_Content_Percent', 0):.1f}%. "
        f"AMR genes detected: {', '.join(other_genes[:18]) or 'none'}. "
        f"Resistance mechanisms: {mech}. "
        f"Resistant to: {', '.join(res_classes) or 'none'}. "
        f"Predict: Is the {gene_name} resistance gene present in this isolate?"
    )


def run_loo_biolinkbert(df, gene: str, y: np.ndarray,
                        gene_cols: list, class_cols: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LOO-CV with BioLinkBERT sequence classifier.
    
    Args:
        df: DataFrame with isolate data
        gene: Target gene column name
        y: True labels array
        gene_cols: List of gene columns
        class_cols: List of class columns
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    prompts = [
        build_clf_prompt(row.to_dict(), gene, gene_cols, class_cols)
        for _, row in df.iterrows()
    ]
    
    loo = LeaveOneOut()
    y_pred, y_prob = [], []
    
    for fold, (tr_idx, te_idx) in enumerate(loo.split(prompts)):
        clf = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            token=HF_TOKEN,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)
        
        # Use DataParallel if multiple GPUs
        if N_GPU > 1:
            clf = nn.DataParallel(clf)
        
        # Class-weighted loss
        pos_w = (len(y) - y.sum()) / (y.sum() + 1e-6)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(pos_w)]).to(DEVICE)
        )
        
        tr_texts = [prompts[i] for i in tr_idx]
        te_texts = [prompts[i] for i in te_idx]
        tr_labs = y[tr_idx].tolist()
        
        tr_enc = tok(tr_texts, truncation=True, max_length=MAX_LENGTH,
                     padding=True, return_tensors="pt")
        te_enc = tok(te_texts, truncation=True, max_length=MAX_LENGTH,
                     padding=True, return_tensors="pt")
        
        loader = DataLoader(
            AMRDataset(tr_enc, tr_labs),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        
        opt = torch.optim.AdamW(clf.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * len(loader)
        )
        
        # Training
        clf.train()
        for _ in range(EPOCHS):
            for batch in loader:
                opt.zero_grad()
                out = clf(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss = loss_fn(logits, batch["labels"].to(DEVICE))
                loss.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                opt.step()
                scheduler.step()
        
        # Evaluation
        clf.eval()
        with torch.no_grad():
            te_inp = {k: v.to(DEVICE) for k, v in te_enc.items()}
            out = clf(**te_inp)
            logits = out.logits if hasattr(out, "logits") else out[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(float(probs[1]))
        
        # Cleanup
        if N_GPU > 1:
            clf = clf.module
        del clf
        torch.cuda.empty_cache()
        
        if (fold + 1) % 10 == 0:
            print(f"    LOO fold {fold+1}/50  "
                  f"acc={accuracy_score(y[:fold+1], y_pred):.3f}")
    
    return np.array(y_pred), np.array(y_prob)
