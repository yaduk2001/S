import os
os.environ['HF_HOME'] = os.path.abspath('./hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('./hf_cache/transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.abspath('./hf_cache/datasets')
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

# 1. Load Yelp Review Full dataset (5k for faster training)
ds = load_dataset('yelp_review_full', split='train')

# 2. Map stars to sentiment label: 0=negative (1-2), 1=neutral (3), 2=positive (4-5)
def map_yelp(example):
    if example['label'] in [0, 1]:  # 1 or 2 stars
        return {'label': 0}
    elif example['label'] == 2:    # 3 stars
        return {'label': 1}
    else:                          # 4 or 5 stars
        return {'label': 2}
ds = ds.map(map_yelp)

# 3. Randomly sample 5,000 reviews
ds = ds.shuffle(seed=42).select(range(5000))

# 4. Use the 'text' field as input
def unify_columns(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in ['text', 'label']])
ds = unify_columns(ds)

# 5. Shuffle and split
dataset = ds.train_test_split(test_size=0.1, seed=42)

# 6. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    texts = batch['text']
    texts = [str(t) if t is not None else "" for t in texts]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)
dataset = dataset.map(tokenize, batched=True)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 7. Model setup
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

print('Data and model ready for training.')

# 8. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    dataloader_num_workers=0,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='none',
    fp16=torch.cuda.is_available(),
)

# 9. Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

# 11. Train
trainer.train()

# 12. Evaluate
results = trainer.evaluate()
print('Evaluation metrics:', results)

# 13. Confusion matrix
preds = trainer.predict(dataset['test'])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:\n', cm)

# 14. Save model
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
print('Model and tokenizer saved to ./sentiment_model') 