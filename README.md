# XLNet Emotion Classification (Fine-Tuning Project)

> End-to-end Transformer fine-tuning pipeline built independently to classify text emotions using XLNet and HuggingFace.

---

## Project Overview

This project is a fully implemented and self-driven fine-tuning pipeline for multi-class emotion classification.

It classifies text into four emotional categories:

- Anger
- Fear
- Joy
- Sadness

The model is fine-tuned from:

xlnet-base-cased

Unlike a guided tutorial implementation, this version was rebuilt independently with:

- Custom preprocessing pipeline
- Manual class balancing
- Stratified train/val/test splitting
- Gradient accumulation
- Mixed precision (fp16)
- Best model checkpoint loading
- Production-style inference pipeline

---

## End-to-End Architecture

CSV Data (3 files)
        ↓
Data Merging
        ↓
Text Cleaning (emoji, URL, mentions removal)
        ↓
Class Balancing (downsampling)
        ↓
Label Encoding
        ↓
Stratified Train / Val / Test Split
        ↓
HuggingFace Dataset Conversion
        ↓
Tokenization (XLNetTokenizer)
        ↓
Fine-Tuning (Trainer API)
        ↓
Evaluation
        ↓
Inference Pipeline

---

## Dataset

Three CSV files were used:

- emotion-labels-train.csv
- emotion-labels-test.csv
- emotion-labels-val.csv

After merging:

- Total samples: 36
- Balanced to 8 samples per class

### Class Distribution After Balancing

| Emotion | Samples |
|----------|----------|
| anger    | 8 |
| fear     | 8 |
| joy      | 8 |
| sadness  | 8 |

Balancing was performed via controlled downsampling to eliminate bias.

---

## Text Preprocessing

Cleaning pipeline includes:

- Emoji removal
- URL removal
- Email & phone removal
- Username (@mention) removal
- Lowercasing
- Whitespace normalization

Example:

```python
def clean_my_text(text):
    text = clean(text, no_emoji=True, no_urls=True, no_emails=True, no_phone_numbers=True, lang='en')
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text
```

---

## Label Encoding

Mapping:

| Label | ID |
|--------|----|
| anger  | 0 |
| fear   | 1 |
| joy    | 2 |
| sadness| 3 |

Encoded using `LabelEncoder`.

---

## Data Splitting Strategy

Stratified splitting ensures class distribution consistency:

- 80% train+val
- 20% test
- Within train+val → 80% train / 20% validation

Final counts:

- Train: 20 samples
- Validation: 5 samples
- Test: 7 samples

---

## Tokenization

Tokenizer used:

```python
XLNetTokenizer.from_pretrained("xlnet-base-cased")
```

Configuration:

- max_length = 128
- padding = "max_length"
- truncation = True

---

## Model Fine-Tuning

Model:

```python
XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels=4
)
```

### Training Configuration Highlights

- 3 epochs
- Batch size: 4
- Gradient accumulation: 4
- Mixed precision (fp16=True)
- Evaluation every epoch
- Save best model
- Load best model at end

This simulates a memory-aware production training setup.

---

## Evaluation Result

After 1 epoch (small dataset):

Validation Accuracy: 0.571

Note:
The dataset is intentionally small for experimentation and reproducibility.
Performance improves significantly with larger datasets.

---

## Inference Example

Using HuggingFace pipeline:

```python
clf = pipeline(
    "text-classification",
    model="./my_fine_tuned_xlnet_emotion",
    tokenizer=tokenizer,
    return_all_scores=True
)
```

Example prediction:

Input:
"I feel so happy today, finally got promoted!"

Output:
[
  {"label": "anger", "score": 0.18},
  {"label": "fear", "score": 0.15},
  {"label": "joy", "score": 0.37},
  {"label": "sadness", "score": 0.28}
]

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- Scikit-learn
- Evaluate
- Matplotlib

---

## Future Improvements

- Larger dataset
- Add F1-score, recall, precision
- Hyperparameter search
- Add experiment tracking (W&B)
- Add confusion matrix visualization
- Compare against BERT / RoBERTa
- Export model to ONNX
