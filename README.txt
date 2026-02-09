# 💙 Milo AI: Multimodal Mental Health Companion

> **"An AI friend that listens to your words and sees your emotions."**

Milo AI is a full-stack therapeutic companion application designed to provide accessible, real-time emotional support. By bridging **Natural Language Processing (NLP)** and **Computer Vision (CV)**, Milo creates a safe, judgment-free space for users to navigate complex emotions.

## 🚀 Key Features

* **💬 Empathetic Chat Engine:** Uses a fine-tuned **RoBERTa** model to classify mental health states (e.g., Anxiety, Suicidal ideation, Depression) and generates therapeutically validated responses.
* **📸 Photo Locket (Computer Vision):** A custom **ResNet-50** model (trained on FER-2013) analyzes facial expressions in real-time to detect mood, logging emotional snapshots into a visual memory gallery.
* **📊 Insight Dashboard:** Visualizes emotional trends over time to help users understand their mental well-being.
* **📱 Mobile-First Design:** Fully accessible on smartphones via secure **Ngrok** tunneling for on-the-go support.

## 🛠️ Tech Stack

* **Framework:** Streamlit (Python)
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Models:**
    * *NLP:* `roberta-base-go_emotions` (Fine-tuned for text analysis)
    * *Vision:* `microsoft/resnet-50` (Fine-tuned on FER-2013 with Data Augmentation)
* **Deployment:** Ngrok (Tunneling)

---

## 🧠 How It Works

1.  **Text Analysis:** The user chats with Milo. The NLP model detects the underlying emotion (e.g., "Fear") and selects a response strategy (e.g., "Validation" or "Grounding Technique").
2.  **Visual Analysis:** The user snaps a photo. The CNN processes the image, classifies the facial expression, and logs the "Emotional Memory" into the digital locket.
3.  **Safety Rails:** Specific triggers (e.g., self-harm content) immediately activate crisis intervention protocols.

---
*Built with 💙 for the "MindMate" Project.*