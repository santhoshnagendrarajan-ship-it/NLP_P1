# =============================================================================
#  Sentiment Analyzer — Streamlit App
#  Artifacts loaded from local paths (no UI upload needed).
#
#  Place these files in the SAME folder as this script:
#    cs_lstm_model.pth    — torch checkpoint
#    lstmencoder.pickle   — sklearn LabelEncoder
#
#  Or update the paths below to wherever your files are saved.
#
#  Run:  streamlit run appv2.py
# =============================================================================

import os, re, pickle
import nltk, torch, torch.nn as nn, numpy as np, streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ── Artifact paths — update these if files are in a different folder ──
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "cs_lstm_model.pth")
ENCODER_PATH = os.path.join(BASE_DIR, "lstmencoder.pickle")

# ── Constants — must match nn_2_1.ipynb ───────────────────────────────
MAX_LEN = 20

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    /* ── center content ── */
    .block-container { max-width: 720px; padding-top: 2rem; }

    /* ── result card ── */
    .result-card {
        padding: 28px 20px; border-radius: 16px;
        text-align: center; font-size: 26px;
        font-weight: 700; margin: 24px 0 8px;
        letter-spacing: 0.5px;
    }
    .positive { background:#d4edda; color:#155724; border:2px solid #28a745; }
    .negative { background:#f8d7da; color:#721c24; border:2px solid #dc3545; }
    .neutral  { background:#fff3cd; color:#856404; border:2px solid #ffc107; }

    /* ── probability row ── */
    .prob-label { font-size:15px; font-weight:600; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ── NLTK downloads ────────────────────────────────────────────────────
@st.cache_resource
def _download_nltk():
    for pkg in ['punkt', 'wordnet', 'stopwords', 'punkt_tab']:
        nltk.download(pkg, quiet=True)
_download_nltk()

# ── Model definition — exact copy from nn_2_1.ipynb ──────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_classes=3, num_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention  = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes)
        )

    def attention_pool(self, lstm_out):
        weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        return (lstm_out * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x, lengths=None):
        emb = self.embed_dropout(self.embedding(x))
        if lengths is not None:
            emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(emb)
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        return self.classifier(self.attention_pool(out))

# ── Load artifacts from disk (cached — runs only once) ────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────
    safe_globals = [np.ndarray]
    try:
        if hasattr(np, '_core'):
            safe_globals.append(np._core.multiarray._reconstruct)
    except Exception:
        pass
    try:
        with torch.serialization.safe_globals(safe_globals):
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    word2idx        = ckpt.get('word2idx', {})
    encoder_classes = ckpt.get('encoder_classes', None)

    if not word2idx:
        st.error("❌ Checkpoint missing 'word2idx'. Re-save your model with the correct format.")
        st.stop()

    # ── Load label encoder ────────────────────────────────────────────
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        if encoder_classes:
            class _Enc:
                def __init__(self, c): self.classes_ = np.array(c)
                def inverse_transform(self, i): return [self.classes_[int(x)] for x in i]
            encoder = _Enc(encoder_classes)
        else:
            st.error(f"❌ Encoder file not found at: {ENCODER_PATH}")
            st.stop()

    # ── Build model and load weights ──────────────────────────────────
    model = SentimentLSTM(
        vocab_size=len(word2idx), embed_dim=128, hidden_dim=256,
        num_classes=len(encoder.classes_), num_layers=2, dropout=0.5
    ).to(device)

    state = ckpt.get('model_state_dict') or ckpt.get('state_dict')
    if state is None:
        st.error("❌ Checkpoint missing 'model_state_dict'.")
        st.stop()

    model.load_state_dict(state)
    model.eval()
    return model, word2idx, encoder, device

# ── Preprocessing — exact match to nn_2_1.ipynb ───────────────────────
@st.cache_resource
def _nlp_tools():
    return set(stopwords.words("english")), {"not", "no", "never"}, WordNetLemmatizer()

_stop, _neg, _lem = _nlp_tools()

def clean_text(text):
    text  = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = nltk.word_tokenize(text)
    return " ".join(
        w if w in _neg else _lem.lemmatize(w)
        for w in words if w in _neg or w not in _stop
    )

def encode_text(text, word2idx):
    seq = [word2idx.get(w, 1) for w in text.split()]
    seq = (seq + [0] * MAX_LEN)[:MAX_LEN]
    return torch.tensor(seq, dtype=torch.long)

# ── Inference ─────────────────────────────────────────────────────────
def predict(raw_text, model, word2idx, encoder, device):
    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        return None, None, None
    enc     = encode_text(cleaned, word2idx).unsqueeze(0).to(device)
    lengths = (enc != 0).sum(dim=1).cpu().clamp(min=1)
    with torch.no_grad():
        probs     = torch.softmax(model(enc, lengths), dim=1).squeeze()
        label_idx = torch.argmax(probs).item()
    label      = encoder.inverse_transform([label_idx])[0]
    confidence = probs[label_idx].item()
    all_probs  = dict(zip(encoder.classes_.tolist(), probs.cpu().numpy().tolist()))
    return label, confidence, all_probs

# ── Load model once at startup ────────────────────────────────────────
model, word2idx, encoder, device = load_model()

# ═════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════
st.title("🧠 Sentiment Analyzer")
st.markdown("Enter any customer feedback below and click **Predict** to classify its sentiment.")
st.markdown("---")

user_input = st.text_area(
    label="Customer Feedback",
    placeholder="e.g. The service was excellent and delivery was super fast!",
    height=140,
    label_visibility="collapsed"
)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🔍  Predict Sentiment", use_container_width=True)

if predict_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, all_probs = predict(user_input, model, word2idx, encoder, device)

        if label is None:
            st.error("Could not extract meaningful tokens. Try a longer or clearer sentence.")
        else:
            emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
            emoji     = emoji_map.get(label, "🔵")

            # ── Result card ───────────────────────────────────────────
            st.markdown(
                f'<div class="result-card {label.lower()}">'
                f'{emoji}&nbsp; {label}'
                f'<span style="font-size:16px; font-weight:400; margin-left:16px;">'
                f'Confidence: {confidence:.1%}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── Probability bars ──────────────────────────────────────
            st.markdown("#### 📊 Class Probabilities")
            for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                st.markdown(f"<p class='prob-label'>{cls} — {prob:.1%}</p>",
                            unsafe_allow_html=True)
                st.progress(float(prob))

st.markdown("---")
st.caption("Bidirectional LSTM · Attention Pooling · PyTorch · Streamlit")