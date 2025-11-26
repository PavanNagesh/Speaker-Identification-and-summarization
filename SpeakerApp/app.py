import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import whisper
import google.generativeai as genai
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os
from model import EmbeddingExtractor 

# --- CONFIGURATION ---
MODEL_PATH = "best_gru_encoder_model.pth"
NUM_SPEAKERS_TRAINED = 251 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! PASTE YOUR GOOGLE API KEY HERE !!!
# (Replace 'PASTE_YOUR_KEY_HERE' with your actual key starting with AIza...)
GOOG_API_KEY = "AIzaSyCpdigOk69FwZG6v1Ipq1jhySmEazLwmas" 

# Configure Gemini
if GOOG_API_KEY == "PASTE_YOUR_KEY_HERE":
    st.error("⚠️ Please edit app.py and paste your Google Gemini API Key in line 22!")
else:
    genai.configure(api_key=GOOG_API_KEY)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Custom Diarization Model
    diar_model = EmbeddingExtractor(num_speakers=NUM_SPEAKERS_TRAINED)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        diar_model.load_state_dict(state_dict, strict=False)
        diar_model.to(DEVICE)
        diar_model.eval()
    except Exception as e:
        st.error(f"Error loading custom model: {e}")
        return None, None, None

    # 2. Whisper (Transcriber)
    transcriber = whisper.load_model("base")

    # 3. Gemini Model (Summarizer)
    # CHANGED: Using a valid model from your list
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    return diar_model, transcriber, gemini_model

# --- HELPER FUNCTIONS ---
def process_audio(audio_path, model):
    y, sr = librosa.load(audio_path, sr=16000)
    window_len = int(3.0 * sr)
    hop_len = int(0.3 * sr)
    embeddings = []
    
    num_steps = int(np.ceil((len(y) - window_len) / hop_len)) + 1
    
    for i in range(num_steps):
        start = i * hop_len
        end = start + window_len
        chunk = y[start:end]
        if len(chunk) < window_len:
            chunk = np.pad(chunk, (0, window_len - len(chunk)), 'constant')
            
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=80)
        S_db = librosa.power_to_db(S, ref=np.max)
        if S_db.shape[1] > 94: S_db = S_db[:, :94]
        elif S_db.shape[1] < 94: S_db = np.pad(S_db, ((0, 0), (0, 94 - S_db.shape[1])), 'constant')
             
        tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(tensor)
        embeddings.append(emb.cpu().numpy().squeeze())

    return np.array(embeddings), sr, hop_len, y

def cluster_embeddings(embeddings, num_speakers):
    clustering = SpectralClustering(n_clusters=num_speakers, affinity='cosine', assign_labels='kmeans', random_state=42)
    labels = clustering.fit_predict(embeddings)
    labels = medfilt(labels, kernel_size=5).astype(int)
    return labels

def plot_diarization(y, sr, labels, hop_len):
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='lightgray', alpha=0.7)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, label in enumerate(labels):
        start_time = (i * hop_len) / sr
        end_time = ((i + 1) * hop_len) / sr
        color = colors[label % len(colors)]
        ax.axvspan(start_time, end_time, color=color, alpha=0.3, lw=0)
    ax.set_title("Diarization Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(y)/sr)
    
    unique_labels = np.unique(labels)
    legend_patches = [plt.Rectangle((0,0),1,1, color=colors[l % len(colors)], alpha=0.3) for l in unique_labels]
    ax.legend(legend_patches, [f"Speaker {l+1}" for l in unique_labels], loc='upper right')
    
    return fig

# --- APP INTERFACE ---
st.title("🎙️ End-To-End Identification & Summarization")
st.markdown("Using **Custom CNN-GRU** + **Whisper** + **Google Gemini 2.0**")

# Sidebar
st.sidebar.header("Debug / Proof")
show_internals = st.sidebar.checkbox("Show Model Internals")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "flac", "mp3"])

if uploaded_file:
    with open("temp_input.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio("temp_input.wav")
    
    num_people = st.slider("Number of Speakers", min_value=1, max_value=5, value=2)
    
    if st.button("Analyze Audio"):
        
        # 1. Load Models
        with st.spinner("Loading AI Models..."):
            diar_model, transcriber, gemini_model = load_models()

        if diar_model and gemini_model:
            status = st.status("Processing Audio...", expanded=True)
            
            # --- STEP A: DIARIZATION ---
            status.write("🔍 1. Extracting Speaker Embeddings...")
            embeddings, sr, hop_len, y = process_audio("temp_input.wav", diar_model)
            
            status.write("🧩 2. Clustering Speakers...")
            speaker_labels = cluster_embeddings(embeddings, num_speakers=num_people)
            
            # --- STEP B: TRANSCRIPTION ---
            status.write("📝 3. Transcribing Text...")
            result = transcriber.transcribe("temp_input.wav")
            segments = result['segments']
            
            # --- STEP C: BUILD CONVERSATION SCRIPT ---
            status.write("🔗 4. Mapping Speakers to Text...")
            
            diarized_transcript = []
            conversation_string = "" 
            
            for seg in segments:
                start_t = seg['start']
                end_t = seg['end']
                text = seg['text'].strip()
                
                start_idx = int(start_t * sr / hop_len)
                end_idx = int(end_t * sr / hop_len)
                
                if start_idx < len(speaker_labels):
                    segment_labels = speaker_labels[start_idx:end_idx]
                    if len(segment_labels) > 0:
                        speaker_id = np.bincount(segment_labels).argmax()
                        speaker_name = f"Speaker {speaker_id + 1}" 
                    else: speaker_name = "Unknown"
                else: speaker_name = "Unknown"
                
                diarized_transcript.append({"speaker": speaker_name, "text": text})
                conversation_string += f"{speaker_name}: {text}\n"
            
            # --- STEP D: SUMMARIZATION (Using Gemini 2.0) ---
            status.write("🧠 5. Generating Summary with Gemini 2.0...")
            
            prompt = f"""
            You are an expert meeting secretary. 
            Here is a transcript of a conversation where the speakers are identified (Speaker 1, Speaker 2).
            
            Please write a clear, concise summary of this conversation.
            Make sure to correctly attribute what each speaker said.
            
            TRANSCRIPT:
            {conversation_string}
            
            SUMMARY:
            """
            
            try:
                response = gemini_model.generate_content(prompt)
                final_summary = response.text
            except Exception as e:
                final_summary = f"Error generating summary: {e}"
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # --- DISPLAY RESULTS ---
            st.subheader("📊 Speaker Waveform")
            fig_waveform = plot_diarization(y, sr, speaker_labels, hop_len)
            st.pyplot(fig_waveform)
            
            if show_internals:
                st.divider()
                st.subheader("🛠️ Model Internals")
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embeddings)
                fig_pca, ax_pca = plt.subplots()
                ax_pca.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=speaker_labels, cmap='viridis', alpha=0.6)
                st.pyplot(fig_pca)

            st.divider()
            
            st.header("🗣️ Speaker-Identified Transcript")
            for line in diarized_transcript:
                spk = line['speaker']
                txt = line['text']
                if "Speaker 1" in spk: st.info(f"**{spk}:** {txt}")
                elif "Speaker 2" in spk: st.warning(f"**{spk}:** {txt}")
                else: st.write(f"**{spk}:** {txt}")

            st.divider()

            st.header("📝 Meeting Summary (by Gemini)")
            st.success(final_summary)