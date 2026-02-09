import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import torch
import random
import os
import json
import numpy as np
import base64 
from PIL import Image

# --- 0. MOBILE ACCESS (NGROK) ---
from pyngrok import ngrok

NGROK_KEY = "YOUR_NGROK_KEY"
ngrok.set_auth_token(NGROK_KEY)

@st.cache_resource
def get_mobile_link():
    ngrok.kill()
    try:
        tunnel = ngrok.connect(8501, "http")
        return tunnel.public_url
    except Exception as e:
        return None

public_url = get_mobile_link()

# --- NEW IMPORTS FOR VISION ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification
from torchvision import transforms

# --- 1. CONFIGURATION ---
PAGE_TITLE = "Milo AI"
PAGE_ICON = "💙"
LOG_FILE = "milo_memory.csv"
PHOTO_DIR = "milo_photos"          # <--- NEW
PHOTO_LOG = "milo_photo_log.csv"   # <--- NEW

# Ensure directories exist
os.makedirs(PHOTO_DIR, exist_ok=True)

# 8 Emotion Classes
CLASS_MAPPING = [
    "Anxiety", "Depression", "Suicidal", "Stress", 
    "Fear", "Sad", "Happy", "Normal"
]

# FER-2013 Labels (Vision Model)
FACE_LABELS = [
    "angry", "disgust", "fear", "happy", 
    "neutral", "sad", "surprise"
]

# Mapping: Vision -> Milo
VISION_TO_MILO = {
    "angry": "Stress",
    "disgust": "Stress",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Normal",
    "sad": "Sad",
    "surprise": "Happy" 
}

# --- 2. THERAPEUTIC RESPONSE ENGINE ---
SUPPORT_DB = {
    "Anxiety": {
        "openers": [
            "I can hear that things feel overwhelming right now.",
            "It sounds like your mind is racing a mile a minute.",
            "I sense a lot of worry in your words, and that's okay.",
            "Take a deep breath with me. I'm right here."
        ],
        "validators": [
            "It is completely valid to feel scared when things feel uncertain.",
            "Anxiety has a way of lying to us, making everything look bigger than it is.",
            "You are not 'crazy' for feeling this way; you are just carrying a lot.",
        ],
        "closers": [
            "We will get through this one moment at a time.",
            "You are safe here with me.",
            "You are stronger than this feeling."
        ]
    },
    "Depression": {
        "openers": [
            "I hear how heavy your heart is today.",
            "It sounds like you're walking through a fog right now.",
            "I'm so sorry things are this hard. I'm listening."
        ],
        "validators": [
            "Please be gentle with yourself; you are doing the best you can.",
            "This feeling is not a reflection of your worth. You are valuable.",
            "It's okay to not be okay right now. Rest is not a failure.",
        ],
        "closers": [
            "I am sending you so much love.",
            "I'm not going anywhere. I'm right here.",
            "Just keep breathing. That is enough for today."
        ]
    },
    "Suicidal": {
        "openers": ["Please stay.", "You are important."], 
        "validators": ["The world needs you."], 
        "closers": ["Please call 988."]
    },
    "Stress": {
        "openers": [
            "Wow, it sounds like you are carrying the weight of the world.",
            "I can tell you are under so much pressure."
        ],
        "validators": [
            "Remember, you are a human being, not a machine.",
            "You have handled hard things before, but you shouldn't have to do it alone.",
            "It is okay to put down the load for just a minute."
        ],
        "closers": [
            "I believe in your ability to handle this.",
            "Let's focus on just the very next step."
        ]
    },
    "Fear": {
        "openers": [
            "I can tell you're feeling scared right now.",
            "It sounds like something has really shaken you up."
        ],
        "validators": [
            "Fear is your body trying to protect you, but you are safe here.",
            "It takes courage to admit when we are afraid."
        ],
        "closers": [
            "You are not in this alone.",
            "I've got your back."
        ]
    },
    "Sad": {
        "openers": [
            "I'm sorry to hear you're feeling down.",
            "It sounds like a sorrowful day.",
            "I can hear the sadness in your words."
        ],
        "validators": [
            "It is perfectly okay to cry.",
            "Sadness honors what we care about. It's okay to feel it."
        ],
        "closers": [
            "I'm sending you a virtual hug.",
            "I'll be here until the clouds clear."
        ]
    },
    "Happy": {
        "openers": [
            "That is wonderful to hear!",
            "I love seeing you this happy!",
            "Yay! That sounds amazing!"
        ],
        "validators": [
            "You absolutely deserve this moment.",
            "Soak it all in!",
            "It is great to celebrate the wins."
        ],
        "closers": [
            "Keep shining!",
            "Thanks for sharing this joy with me."
        ]
    },
    "Normal": {
        "openers": ["I'm glad to hear from you!", "It sounds like things are steady."],
        "validators": ["I love seeing you like this.", "It's great to just chat."],
        "closers": ["I'm always here if you need me.", "I hope the rest of your day is beautiful."]
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads BOTH Text and Vision models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. Text Model Paths ---
    txt_path_options = [
        "/home/pudds/tcu-programming/MILO/models/roberta-mental-health", # Specific absolute path
        "../models/roberta-mental-health",
        "models/roberta-mental-health"
    ]
    
    txt_tokenizer, txt_model = None, None
    for path in txt_path_options:
        if os.path.exists(path):
            try:
                txt_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                txt_model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
                txt_model.to(device)
                print(f"✅ Text Model loaded from {path}")
                break
            except: continue

    # --- B. Vision Model Paths ---
    face_path_options = [
        "/home/pudds/tcu-programming/MILO/models/milo_face_model", # Specific absolute path
        "../models/milo_face_model",
        "models/milo_face_model"
    ]
    
    vis_model = None
    for path in face_path_options:
        if os.path.exists(path):
            try:
                vis_model = AutoModelForImageClassification.from_pretrained(
                    path, 
                    num_labels=7,
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                )
                vis_model.to(device)
                vis_model.eval()
                print(f"✅ Vision Model loaded from {path}")
                break
            except: continue
            
    return txt_tokenizer, txt_model, vis_model, device

tokenizer, text_model, face_model, device = load_models()

# --- 4. LOGIC FUNCTIONS ---

def analyze_text(text):
    if not tokenizer or not text_model: return "Normal", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device
    with torch.no_grad():
        logits = text_model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu() # Move back to CPU
    suicidal_idx = CLASS_MAPPING.index("Suicidal")
    if probs[suicidal_idx] > 0.4: return "Suicidal", probs[suicidal_idx].item()
    max_prob, max_idx = torch.max(probs, dim=0)
    if max_prob < 0.35: return "Normal", max_prob.item()
    return CLASS_MAPPING[max_idx], max_prob.item()

def analyze_image(image_file):
    """New Function for Image Analysis"""
    if not face_model: return None, None
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = face_model(img_tensor).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_class_id = torch.max(probs, 1)
        raw_emotion = FACE_LABELS[top_class_id.item()]
        return VISION_TO_MILO.get(raw_emotion, "Normal"), f"{raw_emotion} ({top_prob.item():.2f})"
    except Exception as e:
        print(f"Image Error: {e}")
        return None, None

def generate_response(mood, text_input=""):
    """Constructs response. text_input is optional for images."""
    if mood == "Suicidal":
        return "I am hearing deeply serious pain in your words. Please, I want you to stay safe. You are incredibly important. Please reach out to a human or call 988. I am here, but I want you safe."

    reflection = ""
    text_lower = text_input.lower()
    if "tired" in text_lower: reflection = " It sounds like you are exhausted. "
    elif "alone" in text_lower: reflection = " I know it feels lonely right now. "
    elif "thanks" in text_lower: reflection = " You are very welcome. "
    
    db = SUPPORT_DB.get(mood, SUPPORT_DB["Normal"])
    return f"{random.choice(db['openers'])}{reflection} {random.choice(db['validators'])} {random.choice(db['closers'])}"

def log_mood(mood):
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["Date", "Mood"]).to_csv(LOG_FILE, index=False)
    new_row = pd.DataFrame([[datetime.now(), mood]], columns=["Date", "Mood"])
    new_row.to_csv(LOG_FILE, mode='a', header=False, index=False)

def save_photo_entry(image_file, mood):
    """Saves photo and logs mood"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    filepath = os.path.join(PHOTO_DIR, filename)
    img = Image.open(image_file)
    img.save(filepath, "JPEG")
    if not os.path.exists(PHOTO_LOG):
        pd.DataFrame(columns=["Timestamp", "Filename", "Mood"]).to_csv(PHOTO_LOG, index=False)
    new_row = pd.DataFrame([[datetime.now(), filename, mood]], columns=["Timestamp", "Filename", "Mood"])
    new_row.to_csv(PHOTO_LOG, mode='a', header=False, index=False)

# --- 5. UI LAYOUT ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .title-text { color: #89CFF0; text-align: center; font-family: sans-serif; }
    .chat-bubble { padding: 20px; border-radius: 15px; margin-bottom: 10px; }
    .user-bubble { background-color: #2b313e; color: #E0E0E0; margin-left: 20%; }
    .milo-bubble { background-color: #1c2333; color: #FFFFFF; margin-right: 20%; border: 1px solid #89CFF0; }
    .big-icon { font-size: 60px; text-align: center; display: block; margin-bottom: 10px; }
    .locket-container { background-color: #1c2333; border-radius: 15px; padding: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #3c4353;}
    .locket-caption { color: #89CFF0; font-size: 0.9em; margin-top: 5px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📱 Mobile Access")
    if public_url:
        st.success(f"**Status: ONLINE**")
        st.markdown(f"[Click to Open on Phone]({public_url})")
        st.caption("Copy this link to your smartphone browser.")
    else:
        st.warning("Tunnel failed. Check internet.")

st.markdown("<div class='big-icon'>💙</div>", unsafe_allow_html=True)
st.markdown(f"<h1 class='title-text'>{PAGE_TITLE}</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📷 Photo Locket", "📊 Insights"])

# --- TAB 1: CHAT ---
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "milo", "content": "Hi there. I'm here for you. How are you feeling today?"}]

    for msg in st.session_state.messages:
        div_class = "user-bubble" if msg["role"] == "user" else "milo-bubble"
        prefix = "You: " if msg["role"] == "user" else "Milo: "
        st.markdown(f"<div class='chat-bubble {div_class}'><b>{prefix}</b><br>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Type here...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        mood, conf = analyze_text(user_input)
        log_mood(mood)
        response_text = generate_response(mood, user_input)
        st.session_state.messages.append({"role": "milo", "content": response_text})
        st.rerun()

# --- TAB 2: PHOTO LOCKET ---
with tab2:
    st.header("Your Photo Locket")
    img_buffer = st.camera_input("Take a photo")
    uploaded = st.file_uploader("Or upload", type=['jpg', 'jpeg', 'png'])
    image_to_process = img_buffer if img_buffer else uploaded

    if image_to_process:
        if "last_processed_img" not in st.session_state or st.session_state.last_processed_img != image_to_process.name:
             with st.spinner("Milo is looking at your photo..."):
                milo_mood, raw_details = analyze_image(image_to_process)
                if milo_mood:
                    image_to_process.seek(0)
                    save_photo_entry(image_to_process, milo_mood)
                    st.success(f"Emotion: **{milo_mood}** [{raw_details}]")
                    # Pass empty string for text_input since this is an image
                    st.info(f"💙 Milo says: {generate_response(milo_mood, '')}")
                    st.session_state.last_processed_img = image_to_process.name
                    st.rerun()

    st.divider()
    # Gallery Logic
    if os.path.exists(PHOTO_LOG):
        df = pd.read_csv(PHOTO_LOG).iloc[::-1] # Reverse order (newest first)
        cols = st.columns(3)
        for idx, row in df.iterrows():
            filepath = os.path.join(PHOTO_DIR, row['Filename'])
            if os.path.exists(filepath):
                with cols[idx % 3]:
                    # READ AND ENCODE IMAGE
                    with open(filepath, "rb") as f:
                        img_bytes = f.read()
                        b64_string = base64.b64encode(img_bytes).decode()
                    
                    st.markdown(f"""
                        <div class="locket-container">
                            <img src="data:image/jpeg;base64,{b64_string}" width="100%" style="border-radius: 10px;">
                            <div class="locket-caption"><b>{row['Mood']}</b><br>{pd.to_datetime(row['Timestamp']).strftime("%b %d")}</div>
                        </div>
                    """, unsafe_allow_html=True)

# --- TAB 3: INSIGHTS ---
with tab3:
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        
        if not df.empty:
            st.markdown("### Your Emotional Spectrum")
            
            domain = ["Anxiety", "Depression", "Suicidal", "Stress", "Fear", "Sad", "Happy", "Normal"]
            range_ = ["#FFA07A", "#708090", "#000000", "#FF4500", "#800080", "#4682B4", "#FFD700", "#98FB98"]

            chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('Mood', sort='-y'),
                y='count()',
                color=alt.Color('Mood', scale=alt.Scale(domain=domain, range=range_))
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.write("Recent Logs:")
            st.dataframe(df.tail(5).sort_index(ascending=False), hide_index=True)
        else:
            st.info("No data yet.")
    else:
        st.info("Start chatting to generate insights.")