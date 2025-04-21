# Cognitive Impairment Detection from Voice

A voice-based application to detect patterns indicative of cognitive decline using audio and speech features.

---

## Final Report Summary

### Insightful Features
- **Audio Features**:
  - Zero Crossing Rate (ZCR): Indicates signal noisiness.
  - Root Mean Square (RMS): Reflects speech energy.
  - Pitch (F0): Captures monotonic speech patterns.
  - Tempo: Used as a proxy for speech rate.
  - Duration: Total speech time, indicating fluency or hesitation.
- **Text Features**:
  - Transcription via Whisper ASR.
  - Linguistic complexity and structure (to be expanded for full clinical integration).

> The **combination of vocal prosody and textual semantics** was the most insightful in surfacing anomalies.

---

### ML Methods Used
- **Feature Extraction**:
  - `librosa` for signal features.
  - OpenAI Whisper ASR for transcription.

- **Dimensionality Reduction**:
  - **UMAP**: Captures non-linear patterns and helps in visualizing clusters effectively.

- **Clustering**:
  - **KMeans**: Applied post-UMAP for grouping similar voice-text embeddings.

- **Anomaly Detection**:
  - **Local Outlier Factor (LOF)**: Highlights outlier behavior in both acoustic and semantic spaces.

---

### Future Steps for Clinical Robustness
- Integrate **clinical datasets** with known cognitive labels for supervised learning.
- Add **longitudinal tracking** to observe decline trends over time.
- Employ **deep learning models** like wav2vec or HuBERT for richer embeddings.
- Incorporate **psycholinguistic metrics** (e.g., word-finding pauses, syntactic variety).
- Ensure **HIPAA/GDPR compliance** for real-world deployment.

---


