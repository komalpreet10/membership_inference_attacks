# Membership Inference Attacks on LLMs and VLMs

> **Note:** This repository contains **only my contributed code** from the project. My contributions include the **Zlib Membership Inference Attack** for Large Language Models (LLMs) and the **Set-Level Temperature Attack** for Vision-Language Models (VLMs). The README summarizes the overall group project for completeness and highlights the best-performing methods, with all methods and results explicitly attributed to their respective contributors.

This project was completed as part of the **COLX 531** and **COLX 585** courses at the **University of British Columbia (UBC)**.

## Team

- Komalpreet
- Nicole Shantz
- Selene Morales
- Shiao-li Green

---

# Overview

This project investigates **Membership Inference Attacks (MIAs)** on both **Large Language Models (LLMs)** and **Vision-Language Models (VLMs)**. The goal is to determine whether a given sample was part of a model's training dataset using several inference strategies, ranging from simple loss-based attacks to supervised classifiers.

For **LLMs**, the strongest-performing approach was the **XGBoost Meta-Classifier** *(Shiao-li Green)*, which achieved an **AUC of 0.8267**.

For **VLMs**, the **CLIP-Based Image-Text Alignment Attack** *(Nicole Shantz)* achieved the best performance with an **AUC of 0.783**, outperforming the baseline and the Set-Level Temperature Attack.

---

# Methods

## Text Models (LLMs)

| Method | AUC | TPR@FPR=0.1 | Contributor |
|---------|----:|------------:|-------------|
| Zlib | 0.6208 | 0.2118 | **Komalpreet** |
| **XGBoost Meta-Classifier** | **0.8267** | **0.4412** | **Shiao-li Green** |

### Methods

- **LOSS** *(Shiao-li Green)* — Computes the average cross-entropy loss across all tokens in a sequence.
- **Zlib** *(Komalpreet)* — Normalizes sequence loss using the compressed length of the input text to reduce text-length bias.
- **Min-k% (Hard Tokens)** *(Selene Morales)* — Uses the highest-loss (low-confidence) tokens to improve membership inference.
- **XGBoost Meta-Classifier** *(Shiao-li Green)* — Combines multiple token-level loss statistics and cross-model features into an XGBoost classifier trained to distinguish members from non-members.

---

## Vision-Language Models (VLMs)

| Method | AUC | TPR@FPR=0.1 | Contributor |
|---------|----:|------------:|-------------|
| Set-Level Temperature Attack | 0.5784 | 0.1560 | **Komalpreet** |
| **CLIP-Based Image-Text Alignment Attack** | **0.783** | **0.257** | **Nicole Shantz** |

### Methods

- **Set-Level Temperature Attack** *(Komalpreet)* — Groups samples into sets of five, queries the model using low (T = 0.1) and high (T = 1.6) temperatures, and measures semantic similarity using **all-mpnet-base-v2** embeddings.
- **Contrastive Pair Margin** *(Selene Morales)* — Compares correct image-caption pairs with mismatched pairs using contrastive margin-based membership scores.
- **Feature Classifier** *(Selene Morales)* — Learns membership inference using supervised classification over contrastive features extracted from image-caption pairs.
- **ICIMIA** *(Shiao-li Green)* — Applies Gaussian blur to images and measures the change in model confidence as a membership signal.
- **CLIP-Based Image-Text Alignment Attack** *(Nicole Shantz)* — Computes cosine similarity between CLIP image and text embeddings to infer membership without requiring access to the target model.

---

# Datasets

## Text Dataset (COLX 531)

Clinical discharge summaries.

| Split | Samples |
|--------|--------:|
| Train | 50,000 |
| Validation | 10,000 |
| Test | 15,000 |

---

## Vision-Language Dataset (COLX 585)

Image-caption pairs.

| Split | Samples |
|--------|--------:|
| Train | 6,000 |
| Validation | 1,200 |
| Test | 6,000 |

**Note:** The VLM dataset is imbalanced with approximately a **3:1 ratio** of non-members to members.

---

# Installation

```bash
pip install torch transformers datasets xgboost scikit-learn pandas tqdm
```

---

# References

- Hu et al. (2025). *Set-Level Temperature Attack.*
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).*
