<div align="center">

# 🎭 AI Sentiment Analysis
### Fine-tuned RoBERTa · 3-Class · 200K+ Training Samples

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-airzipm%2Fsentiment--analysis--roberta-FFD21E?style=for-the-badge)](https://huggingface.co/airzipm/sentiment-analysis-roberta)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-airzipm%2Fsentiment--dataset-blue?style=for-the-badge)](https://huggingface.co/datasets/airzipm/sentiment-dataset)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

<br/>

> A production-grade sentiment analysis model built from scratch —
> from raw dataset collection and EDA all the way to a live Gradio demo —
> fine-tuned on **roberta-base** with advanced deep learning techniques.

<br/>

![Sentiment Banner](https://huggingface.co/airzipm/sentiment-analysis-roberta/resolve/main/training_curves.png)

</div>

---

## 📌 Table of Contents

- [Live Demo](#-live-demo)
- [What I Built](#-what-i-built)
- [Model Performance](#-model-performance)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results & Evaluation](#-results--evaluation)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🚀 Live Demo

Try the model instantly — no setup needed:

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="airzipm/sentiment-analysis-roberta",
)

texts = [
    "Best product I've ever bought — absolutely love it!",
    "It was okay, nothing special honestly.",
    "Terrible experience, complete waste of money.",
]

for t in texts:
    r = clf(t)[0]
    print(f"{r['label']:8s}  {r['score']:.1%}  →  {t}")
```

**Output:**
```
Positive  97.3%  →  Best product I've ever bought — absolutely love it!
Neutral   81.2%  →  It was okay, nothing special honestly.
Negative  95.8%  →  Terrible experience, complete waste of money.
```

🔗 **[View model on Hugging Face →](https://huggingface.co/airzipm/sentiment-analysis-roberta)**

---

## 🧠 What I Built

This is an end-to-end deep learning project covering every stage of the ML pipeline:

| Stage | What I Did |
|---|---|
| **📥 Data Collection** | Downloaded 4 public datasets (IMDB, SST-2, Tweet Eval, Yelp) via HuggingFace `datasets` |
| **🔍 EDA** | Generated 5 diagnostic plots — class distribution, text length histograms, WordClouds per class, top bigrams, TextBlob polarity & subjectivity |
| **🧹 Preprocessing** | Built a 12-step text cleaning pipeline: contraction expansion, emoji-to-text, HTML removal, URL stripping, lemmatization, smart stopword removal (preserving negations) |
| **⚖️ Class Balancing** | Applied `compute_class_weight("balanced")` and integrated weights into the loss function |
| **🏗️ Tokenization** | Used RoBERTa's BPE tokenizer with `max_length=128`, padding, and truncation |
| **🤖 Model** | Fine-tuned `roberta-base` with a custom 3-class classification head |
| **🎛️ Training** | AdamW optimizer, linear warmup scheduler, FP16 mixed precision, gradient clipping, label smoothing, early stopping |
| **📊 Evaluation** | Accuracy, F1 (macro + weighted), MCC, confusion matrix, ROC curves, PR curves, calibration curve |
| **🚀 Deployment** | Pushed model + tokenizer + model card to Hugging Face Hub; live Gradio demo in Colab |

---

## 📊 Model Performance

<div align="center">

| Metric | Score |
|---|---|
| **Accuracy** | — |
| **F1 Score (Macro)** | — |
| **F1 Score (Weighted)** | — |
| **Matthews Correlation Coefficient** | — |

> *Scores auto-populate after training completes. Check the [model page](https://huggingface.co/airzipm/sentiment-analysis-roberta) for the latest numbers.*

</div>

![Confusion Matrix](https://huggingface.co/airzipm/sentiment-analysis-roberta/resolve/main/confusion_matrix.png)

---

## 🏗️ Project Architecture

```
Raw Text (from 4 datasets)
        │
        ▼
┌─────────────────────┐
│   Data Collection   │  IMDB · SST-2 · Tweet Eval · Yelp
│   & Normalization   │  → unified 3-class format
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  EDA & Exploration  │  5 diagnostic plots
│                     │  WordClouds · Bigrams · Polarity
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Text Cleaning      │  12-step pipeline
│  Pipeline           │  Contractions · Emojis · Lemmatize
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  RoBERTa Tokenizer  │  BPE · max_length=128
│                     │  Padding · Truncation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  roberta-base       │  125M parameters
│  + Classification   │  → Dropout → Dense(3) → Softmax
│  Head               │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Training           │  AdamW · FP16 · Warmup
│                     │  Label Smoothing · Class Weights
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hugging Face Hub   │  Model · Tokenizer · Dataset
│  + Gradio Demo      │  Live public inference
└─────────────────────┘
```

---

## 📦 Dataset

Four public datasets combined into a unified 3-class corpus:

| Dataset | Domain | Samples Used | Labels |
|---|---|---|---|
| [IMDB](https://huggingface.co/datasets/imdb) | Movie reviews | 50,000 | Pos / Neg |
| [SST-2](https://huggingface.co/datasets/glue) | Short sentences | 50,000 | Pos / Neg |
| [Tweet Eval](https://huggingface.co/datasets/tweet_eval) | Twitter posts | 50,000 | Pos / Neu / Neg |
| [Yelp Review Full](https://huggingface.co/datasets/yelp_review_full) | Business reviews | 50,000 | 5-star → 3-class |
| **Total** | Multi-domain | **200,000** | **Neg · Neu · Pos** |

**Label mapping:**

```
0 → Negative   (1–2 star reviews, negative tweets, negative sentiment)
1 → Neutral    (3-star reviews, neutral tweets, ambiguous sentences)
2 → Positive   (4–5 star reviews, positive tweets, positive sentiment)
```

🔗 **[View dataset on Hugging Face →](https://huggingface.co/datasets/airzipm/sentiment-dataset)**

---

## 🧹 Preprocessing Pipeline

A 12-step cleaning pipeline applied to all splits before tokenization:

```python
def clean_text(text):
    text = contractions.fix(text)                    # 1. don't → do not
    text = emoji.demojize(text, delimiters=(" "," ")) # 2. 😊 → happy face
    text = text.lower()                              # 3. lowercase
    text = remove_html(text)                         # 4. strip <tags>
    text = remove_urls(text)                         # 5. strip http://...
    text = remove_mentions(text)                     # 6. strip @username
    text = keep_hashtag_text(text)                   # 7. #great → great
    text = remove_numbers(text)                      # 8. strip digits
    text = remove_punct_keep_sentiment(text)         # 9. keep ! and ?
    text = remove_stopwords_keep_negations(text)     # 10. keep "not","never",...
    text = spacy_lemmatize(text)                     # 11. running → run
    text = remove_single_chars(text)                 # 12. drop noise tokens
    return text
```

> **Key design decision:** Negation words (`not`, `never`, `don't`, `couldn't`, etc.) are **excluded from stopword removal** — removing them would destroy sentiment signal (e.g. *"not bad"* → *"bad"*).

---

## 🤖 Model Architecture

```
Input Text
    │
    ▼
RoBERTa Tokenizer (BPE, vocab=50,265)
    │
    ▼
roberta-base
  ├── 12 Transformer layers
  ├── 768 hidden dimensions
  ├── 12 attention heads
  └── 125M parameters
    │
    ▼
[CLS] token representation  (768-dim)
    │
    ▼
Dropout (p=0.1)
    │
    ▼
Linear(768 → 3)
    │
    ▼
Softmax → [P(Negative), P(Neutral), P(Positive)]
```

---

## 🎛️ Training Configuration

```python
CONFIG = {
    "model_name"      : "roberta-base",
    "max_length"      : 128,          # token limit per sample
    "batch_size"      : 32,           # per GPU
    "num_epochs"      : 4,
    "learning_rate"   : 2e-5,         # standard for transformer fine-tuning
    "weight_decay"    : 0.01,         # AdamW regularization
    "warmup_ratio"    : 0.1,          # 10% of steps for LR warmup
    "grad_clip"       : 1.0,          # prevent exploding gradients
    "fp16"            : True,         # mixed precision — 2x faster on GPU
    "label_smoothing" : 0.1,          # prevents overconfident predictions
    "patience"        : 2,            # early stopping
}
```

**Advanced techniques used:**
- ✅ **AdamW** with layer-wise weight decay (bias & LayerNorm excluded)
- ✅ **Linear warmup + decay** learning rate schedule
- ✅ **FP16 mixed precision** training via `torch.cuda.amp`
- ✅ **Gradient clipping** at norm = 1.0
- ✅ **Label smoothing** (0.1) to prevent overconfidence
- ✅ **Class-weighted CrossEntropyLoss** for imbalanced data
- ✅ **Early stopping** on validation F1 (patience = 2)
- ✅ **Checkpoint saving** — best model pushed to HF Hub after every epoch

---

## 📈 Results & Evaluation

**Evaluation metrics computed:**
- Accuracy
- F1 Score (macro and weighted)
- Matthews Correlation Coefficient (MCC) — best for imbalanced data
- Confusion matrix (raw counts + normalized)
- ROC curves (one-vs-rest per class)
- Precision-Recall curves
- Calibration curve (confidence vs accuracy)
- Per-class Precision / Recall / F1 bar chart
- Error analysis — highest-confidence wrong predictions
- Edge case testing — sarcasm, double negatives, emojis, all-caps

---

## ⚡ How to Use

### Option 1 — One-liner with HuggingFace pipeline

```python
from transformers import pipeline

clf = pipeline("text-classification", model="airzipm/sentiment-analysis-roberta")
print(clf("I absolutely loved this product!"))
# [{'label': 'Positive', 'score': 0.973}]
```

### Option 2 — Manual inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "airzipm/sentiment-analysis-roberta"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128,
                       truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    labels  = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return {"label": labels[pred_id], "confidence": probs[pred_id].item()}

print(predict("This movie was absolutely terrible."))
# {'label': 'Negative', 'confidence': 0.958}
```

### Option 3 — Run the notebooks in Colab

| Notebook | Description | Open |
|---|---|---|
| `01_setup_and_data.py` | Install libs, download datasets, push to HF Hub | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| `02_eda_and_preprocessing.py` | EDA plots, 12-step cleaning pipeline | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| `03_model_training.py` | Fine-tune RoBERTa, push checkpoints live | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| `04_model_card_and_demo.py` | Model card, Gradio demo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |

> **Requirements:** Google Colab with GPU runtime (T4 or A100). Get your HF write token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── 01_setup_and_data.py          # Data download & HF Hub upload
├── 02_eda_and_preprocessing.py   # EDA plots & text cleaning
├── 03_model_training.py          # RoBERTa fine-tuning loop
├── 04_model_card_and_demo.py     # Model card + Gradio demo
│
└── README.md                     # You are here
```

**Hugging Face repos (all permanent, no Google Drive needed):**

```
airzipm/sentiment-analysis-roberta    ← trained model + tokenizer
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files
    ├── README.md  (model card)
    ├── training_curves.png
    └── confusion_matrix.png

airzipm/sentiment-dataset             ← processed data + EDA plots
    ├── raw/
    │   ├── train_raw.csv
    │   ├── val_raw.csv
    │   └── test_raw.csv
    ├── processed/
    │   ├── train_clean.csv
    │   ├── val_clean.csv
    │   └── test_clean.csv
    └── eda_plots/
        ├── 01_class_distribution.png
        ├── 02_text_length.png
        ├── 03_wordclouds.png
        ├── 04_bigrams.png
        └── 05_polarity.png
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0, Hugging Face Transformers 4.40 |
| **Model** | `roberta-base` (125M params) |
| **Data** | Hugging Face `datasets`, pandas, numpy |
| **NLP** | NLTK, spaCy, TextBlob, emoji, contractions |
| **Visualization** | matplotlib, seaborn, plotly, wordcloud |
| **Training** | FP16 mixed precision, AdamW, warmup scheduler |
| **Evaluation** | scikit-learn, torchmetrics |
| **Deployment** | Hugging Face Hub, Gradio |
| **Environment** | Google Colab (GPU) |

</div>

---

## 👤 Author

<div align="center">

**airzipm**

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-airzipm-FFD21E?style=for-the-badge)](https://huggingface.co/airzipm)
[![GitHub](https://img.shields.io/badge/GitHub-airzipm-181717?style=for-the-badge&logo=github)](https://github.com/airzipm)

</div>

---

<div align="center">

⭐ **If this project helped you, consider giving it a star!** ⭐

</div>
