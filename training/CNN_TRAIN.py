import os
import torch
import numpy as np
import pandas as pd 
from datasets import load_dataset
from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    Compose, Normalize, Resize, ToTensor, Grayscale
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Optional: HF Login (Uncomment if needed)
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN" 
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# --- 1. CONFIGURATION ---
MODEL_ID = "microsoft/resnet-50"
DATASET_ID = "AutumnQiu/fer2013"
OUTPUT_DIR = "../models/milo_face_model"
REPORT_FILE = "training_report.csv"

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 50

# FER-2013 Label Map (Crucial for reading the report)
LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# --- 2. DATA PREPARATION ---
print(f"⬇️ Loading dataset: {DATASET_ID}...")
dataset = load_dataset(DATASET_ID)

# Rename to avoid conflict with imported module
data_transform = Compose([
    Grayscale(num_output_channels=3), 
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_fn(examples):
    # Ensure images are RGB (ResNet requires 3 channels)
    examples["pixel_values"] = [data_transform(img.convert("RGB")) for img in examples["image"]]
    del examples["image"] # Cleanup to prevent Trainer crash
    return examples

print("⚙️ Applying transforms...")
dataset = dataset.with_transform(transform_fn)

# Split Validation
if "validation" not in dataset:
    print("✂️ Splitting train/test...")
    split = dataset["train"].train_test_split(test_size=0.1)
    train_ds = split["train"]
    val_ds = split["test"]
else:
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

# --- 3. METRICS ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }

# --- 4. MODEL & TRAINER ---
print(f"🧠 Loading Model: {MODEL_ID}...")
# Map labels to the config so the model knows '0' means 'angry'
model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=7,
    id2label=LABELS,
    label2id={v: k for k, v in LABELS.items()},
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

# --- 5. EXECUTION ---
print("🚀 Starting Training...")
try:
    trainer.train()
    
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    
    # --- 6. DEBUGGING & REPORT GENERATION (NEW) ---
    print("\n📊 Generating Debug Report...")
    
    # 1. Get predictions on the validation set
    predictions_output = trainer.predict(val_ds)
    preds = np.argmax(predictions_output.predictions, axis=1)
    labels = predictions_output.label_ids
    
    # 2. Print Classification Report to Console
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(labels, preds, target_names=LABELS.values()))
    
    # 3. Create CSV Data
    results = []
    for i in range(len(preds)):
        # Calculate confidence (probability of the predicted class)
        probs = torch.nn.functional.softmax(torch.tensor(predictions_output.predictions[i]), dim=0)
        confidence = probs[preds[i]].item()
        
        results.append({
            "True Label ID": labels[i],
            "True Label Name": LABELS[labels[i]],
            "Predicted ID": preds[i],
            "Predicted Name": LABELS[preds[i]],
            "Confidence": round(confidence, 4),
            "Correct": labels[i] == preds[i]
        })

    # 4. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(REPORT_FILE, index=False)
    print(f"\n✅ Report saved to: {REPORT_FILE}")
    print("Check this file to see if 'Predicted Name' is always 'happy'.")
    
except Exception as e:
    print(f"\n❌ Training Error: {e}")