import os
os.environ['HF_HOME'] = os.path.abspath('./hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('./hf_cache/transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.abspath('./hf_cache/datasets')
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

# Load test data (Amazon Polarity, 1000 samples)
test_ds = load_dataset('amazon_polarity', split='test[:1000]')

def map_amazon(example):
    return {'label': 0 if example['label'] == 0 else 2}
test_ds = test_ds.map(map_amazon)

def combine_text(example):
    return {'text': (example['title'] or '') + ' ' + (example['content'] or '')}
test_ds = test_ds.map(combine_text)

def unify_columns(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in ['text', 'label']])
test_ds = unify_columns(test_ds)

# Load model and tokenizer
model_dir = './sentiment_model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Tokenize test data
batch = test_ds['text']
inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
labels = np.array(test_ds['label'])

# Run inference
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# Calculate metrics
acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
cm = confusion_matrix(labels, preds)

print("\nEvaluation Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm) 