"""
MILO Text Model Training - 8 Classes (Happy/Sad/Fear added)
Fixed for Transformers v4.40+ and Dataset Merging
"""
import os
import pandas as pd
import numpy as np
import torch
import json
from torch.nn import BCEWithLogitsLoss
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding, 
)
from sklearn.metrics import f1_score, accuracy_score

# --- 1. CONFIGURATION ---

# Optional: HF Login (Uncomment if needed)
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN" 
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

BASE_MODEL = "SamLowe/roberta-base-go_emotions"
OUTPUT_DIR = "../models/roberta-mental-health"
# LOCAL_CSV = "../data/Combined Data.csv"    #if you want to add local CVS for training link it's path HERE

# 8 Emotion Classes (Your requested update)
MILO_LABELS = [
    "Anxiety", "Depression", "Suicidal", "Stress", 
    "Fear", "Sad", "Happy", "Normal"
]

label2id = {label: i for i, label in enumerate(MILO_LABELS)}
id2label = {i: label for label, i in label2id.items()}
NUM_LABELS = len(MILO_LABELS)

# Mapping: GoEmotions -> Milo 8 Classes
GO_TO_MILO = {
    "suicidal": "Suicidal", 
    "fear": "Fear", "nervousness": "Anxiety", "worry": "Anxiety", "confusion": "Anxiety",
    "remorse": "Depression", "despair": "Depression", "grief": "Sad", "sadness": "Sad", 
    "disappointment": "Sad", "embarrassment": "Sad",
    "anger": "Stress", "annoyance": "Stress", "disapproval": "Stress", "disgust": "Stress",
    "joy": "Happy", "love": "Happy", "admiration": "Happy", "amusement": "Happy", 
    "excitement": "Happy", "gratitude": "Happy", "pride": "Happy", "optimism": "Happy", 
    "relief": "Happy", "desire": "Happy",
    "neutral": "Normal", "realization": "Normal", "approval": "Normal", 
    "caring": "Normal", "curiosity": "Normal", "surprise": "Normal",
}

# Mapping: Your Local CSV -> Milo 8 Classes
CSV_TO_MILO = {
    "Anxiety": "Anxiety",
    "Depression": "Depression",
    "Suicidal": "Suicidal",
    "Stress": "Stress",
    "Normal": "Normal",
    # (Bipolar/Personality Disorder omitted to keep data clean)
}

# --- 2. DATA PREPARATION ---

# [CRITICAL FIX] Define exact schema to prevent "unexpected type" error during merge
target_features = Features({
    "text": Value("string"),
    "labels": Sequence(Value("float32")) 
})

def get_one_hot(label_list):
    """Creates a float32 one-hot vector."""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for label in label_list:
        if label in label2id:
            vec[label2id[label]] = 1.0
    return vec

# 2.1 Process GoEmotions
print("Loading GoEmotions...")
dataset_go = load_dataset("go_emotions", "simplified")
go_emotion_names = dataset_go["train"].features["labels"].feature.names

def process_go(example):
    active = [go_emotion_names[i] for i in example["labels"]]
    mapped_labels = []
    for emo in active:
        if emo in GO_TO_MILO:
            mapped_labels.append(GO_TO_MILO[emo])
    
    if not mapped_labels:
        mapped_labels.append("Normal")
        
    return {"labels": get_one_hot(mapped_labels), "text": example["text"]}

# Apply mapping and FORCE the schema
cols_to_remove = [c for c in dataset_go["train"].column_names if c not in ["text", "labels"]]
dataset_go = dataset_go.map(
    process_go, 
    remove_columns=cols_to_remove,
    features=target_features
)

# 2.2 Process Local CSV
print(f"Loading {LOCAL_CSV}...")
if os.path.exists(LOCAL_CSV):
    df = pd.read_csv(LOCAL_CSV)
    df = df.dropna(subset=['statement', 'status'])
    df = df[df['status'].isin(CSV_TO_MILO.keys())]
    
    texts = df['statement'].astype(str).tolist()
    labels = []
    for status in df['status']:
        target_label = CSV_TO_MILO[status]
        labels.append(get_one_hot([target_label]))
        
    # Create Dataset with FORCED schema
    dataset_csv = Dataset.from_dict(
        {"text": texts, "labels": labels},
        features=target_features
    )
    
    dataset_csv = dataset_csv.train_test_split(test_size=0.1)
    
    train_ds = concatenate_datasets([dataset_go["train"], dataset_csv["train"]])
    val_ds = concatenate_datasets([dataset_go["validation"], dataset_csv["test"]])
    
    print(f"Merged Data: {len(train_ds)} training samples.")
else:
    print("⚠️ CSV not found. Using only GoEmotions.")
    train_ds = dataset_go["train"]
    val_ds = dataset_go["validation"]

# --- 3. TRAINING SETUP ---

print("Calculating class weights...")
all_labels = np.array(train_ds["labels"])
class_counts = np.sum(all_labels, axis=0)
total_samples = len(all_labels)

pos_weights = (total_samples - class_counts) / np.maximum(class_counts, 1)

# Your specific multipliers
multipliers = {
    "Suicidal": 10.0,
    "Depression": 3.0,
    "Anxiety": 2.0,
    "Fear": 1.5,
    "Stress": 1.5,
    "Sad": 1.2,
    "Happy": 1.0,
    "Normal": 0.5
}
weight_tensor = torch.tensor([
    pos_weights[i] * multipliers[MILO_LABELS[i]] 
    for i in range(NUM_LABELS)
], dtype=torch.float32)

print(f"Weights: {dict(zip(MILO_LABELS, weight_tensor.tolist()))}")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = BCEWithLogitsLoss(pos_weight=weight_tensor.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

# --- 4. EXECUTION ---

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing...")
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Format for PyTorch
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,               
    per_device_train_batch_size=16,   
    gradient_accumulation_steps=2,    
    num_train_epochs=3,               
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    eval_strategy="epoch",
    
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    logging_steps=100,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,     
    # tokenizer=tokenizer,           # Removed this to stop crashes
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save Config for MILO.py
with open(os.path.join(OUTPUT_DIR, "milo_inference_config.json"), "w") as f:
    json.dump({"labels": MILO_LABELS, "threshold": 0.5}, f)

print("✅ Training Complete!")
