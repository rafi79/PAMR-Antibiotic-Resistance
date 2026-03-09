# AMR Clinical Intelligence — BioLinkBERT Edition

Antimicrobial resistance (AMR) prediction system using **michiyasunaga/BioLinkBERT-base**, a 110M parameter BERT model pre-trained on PubMed with citation links. 

## amr-biolinkbert.ipynb
This Files contain Sample Run file in python we have run in kaggle.
## Dataset link in Kaggle
We Use the Dataset from Kaggle- https://www.kaggle.com/datasets/vihaankulkarni/antimicrobial-resistance-dataset

## Why BioLinkBERT for AMR?

✓ **110M params** — fits comfortably on a single T4 GPU (15GB)  
✓ **Citation-aware training** — learns which genes co-appear across thousands of AMR papers automatically  
✓ **Best BLURB benchmark score** of all ~110M biomedical BERTs  
✓ **Fill-mask + seq-classification** — no sacremoses, no fp16 tricks  
✓ **LOO-CV 50 folds** runs in ~8 min on T4 (vs 40 min for BioMedLM)

## Methods

**A. Zero-shot fill-mask scoring** (no training)  
**B. Fine-tuned seq. classifier** (LOO-CV, class-weighted)  
**C. Ensemble A+B** (weighted combination)  
**D. BioLinkBERT citation-aware probes** for explanation WHY

## Project Structure

```
├── config.py              # Configuration and hyperparameters
├── knowledge_base.py      # AMR gene and antibiotic class facts
├── data_loader.py         # Data loading and preprocessing
├── model_loader.py        # BioLinkBERT model initialization
├── zero_shot.py           # Zero-shot fill-mask prediction
├── fine_tuning.py         # Fine-tuned LOO-CV classifier
├── citation_probes.py     # Citation-aware probing
├── report_builder.py      # Explanation generation
├── clinical_report.py     # Full clinical report formatting
├── visualization.py       # Dashboard visualization
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Configuration

Edit `config.py` to customize:
- Model name and HuggingFace token
- Data paths
- Training hyperparameters (epochs, learning rate, batch size)
- Ensemble weights
- Visualization colors

## Outputs

- `amr_biolinkbert_results.csv` — AUC/Acc for all genes × all methods
- `amr_biolinkbert_reports.txt` — Full clinical reports with citation probes
- `amr_biolinkbert_summary.csv` — Per-isolate predictions
- `amr_biolinkbert.png` — 8-panel visualization dashboard

## Features

### Zero-Shot Fill-Mask
Uses BioLinkBERT's pre-trained knowledge to score gene presence without training:
- Builds clinical prompts with [MASK] token
- Compares positive vs negative vocabulary probabilities
- Leverages citation-link pre-training for co-occurrence patterns

### Fine-Tuned Classifier
Leave-one-out cross-validation with sequence classification:
- Class-weighted loss for imbalanced data
- DataParallel support for multi-GPU training
- CosineAnnealingLR for smooth convergence

### Citation-Aware Probes
Unique to BioLinkBERT's citation-link training:
- Probes clinical significance
- Probes co-occurrence patterns
- Probes therapy requirements
- Generates explainable predictions

### Clinical Reports
Comprehensive reports including:
- Isolate metadata and resistance profile
- Prediction with confidence scores
- Citation probe results
- Detected genes and resistance classes
- Clinical reasoning with knowledge base facts
- Antibiogram interpretation
- Treatment recommendations
- Infection control guidance

