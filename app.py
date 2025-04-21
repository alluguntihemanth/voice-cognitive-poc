import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from src.audio_processing import extract_audio_features
from src.text_processing import extract_text_features
from src.feature_engineering import combine_features
from src.model import reduce_and_cluster, anomaly_scores
from src.utils import transcribe_audio

# Page Config
st.set_page_config(page_title="Cognitive Decline Detection", layout="centered")

# Title + Upload Section
st.title("Voice-Based Cognitive Impairment Pattern Detection")
uploaded_files = st.file_uploader("Upload Voice Clips (WAV/MP3)", type=["wav", "mp3"], accept_multiple_files=True)

# Setup session state to avoid duplicate processing
if "prev_files" not in st.session_state:
    st.session_state.prev_files = []

# Check if files are new
new_upload = False
if uploaded_files:
    new_names = [f.name for f in uploaded_files]
    if new_names != st.session_state.prev_files:
        new_upload = True
        st.session_state.prev_files = new_names

if uploaded_files and new_upload:
    # Clean up previously written temp files
    for f in os.listdir():
        if f.startswith("temp_") and (f.endswith(".mp3") or f.endswith(".wav")):
            try:
                os.remove(f)
            except Exception as e:
                st.warning(f"Couldn't delete temp file {f}: {e}")

    all_features = []
    file_names = []
    transcriptions = []

    for file in uploaded_files:
        temp_filename = f"temp_{file.name}"
        with open(temp_filename, "wb") as f:
            f.write(file.read())

        try:
            text = transcribe_audio(temp_filename)
            audio_feat = extract_audio_features(temp_filename)
            text_feat = extract_text_features(text)

            if not audio_feat or len(audio_feat) == 0:
                st.warning(f"Skipping {file.name}: No audio features found.")
                continue

            features = combine_features(audio_feat, text_feat)
            all_features.append(features)
            file_names.append(file.name)
            transcriptions.append(text)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    # Proceed only if features were extracted
    if all_features:
        df = pd.DataFrame(all_features)
        df["filename"] = file_names
        X = df.drop(columns=["filename"])

        if len(X) >= 2:
            X_umap, labels = reduce_and_cluster(X)
            scores, _ = anomaly_scores(X)
        else:
            X_umap = [[0, 0] for _ in range(len(X))]
            labels = [0] * len(X)
            scores = [0] * len(X)

        df["cluster"] = labels
        df["anomaly_score"] = scores

        # 📊 Feature Table
        st.subheader("📊 Feature Overview")
        st.dataframe(df, use_container_width=True)

        # 🔍 UMAP Plot + Anomaly Scores
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🧬 UMAP Clusters")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter([p[0] for p in X_umap], [p[1] for p in X_umap], c=labels, cmap='viridis', s=100)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            st.pyplot(fig)

        with col2:
            st.subheader("⚠️ Anomaly Scores")
            st.bar_chart(df.set_index("filename")["anomaly_score"], height=300)

        # 📜 Transcriptions
        st.subheader("📜 Processed Text from Audio Clips")
        for idx, text in enumerate(transcriptions):
            st.write(f"**{file_names[idx]}**:")
            st.text_area(f"Transcription of {file_names[idx]}", text, height=150, max_chars=500, key=f"transcript_{idx}")

        st.success("✅ Processing Complete. Check clusters, scores, and transcriptions above.")
    else:
        st.error("❌ No valid audio clips processed. Please upload valid WAV/MP3 files.")
elif uploaded_files:
    st.info("ℹ️ Please remove previously uploaded files before uploading new ones.")
