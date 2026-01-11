import io
import os
import time
from pathlib import Path

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from captum.attr import IntegratedGradients, Occlusion
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "Model" / "audio_cnn_bilstm_ravdess.pt"))
RAVDESS_DIR = BASE_DIR / "Ravdess Audio"
CREMA_DIR = BASE_DIR / "AudioWAV"
USE_DATASET_DIRS = os.getenv("USE_DATASET_DIRS", "false").lower() == "true"
FEEDBACK_DIR = BASE_DIR / "feedback"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.csv"
DATASET_STATUS_CSV = BASE_DIR / "dataset_status.csv"

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 128
MAX_FRAMES = 800
RMS_FEATURE = True
MAX_RECORD_SECONDS = 10

emotion_order = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
label_map = {label: idx for idx, label in enumerate(emotion_order)}
idx_to_label = {idx: label for label, idx in label_map.items()}
MODEL_STATUS = "loaded" if MODEL_PATH.exists() else "missing (using uninitialized weights)"


class AudioCNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 256,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, cnn_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(cnn_channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels * len(kernel_sizes),
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        adj_lengths = torch.clamp(lengths.to(inputs.device) - 1, min=0)
        gather_idx = adj_lengths.view(-1, 1, 1).expand(-1, 1, unpacked.size(2))
        last_hidden = unpacked.gather(1, gather_idx).squeeze(1)
        return self.head(last_hidden)


def load_model() -> nn.Module:
    input_dim = N_MELS + (1 if RMS_FEATURE else 0)
    model = AudioCNNBiLSTM(input_dim=input_dim, num_classes=len(label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def ensure_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    return np.mean(waveform, axis=1)


def trim_or_pad_waveform(waveform: np.ndarray, sr: int, max_seconds: int) -> np.ndarray:
    max_samples = int(sr * max_seconds)
    if waveform.shape[0] > max_samples:
        return waveform[:max_samples]
    return waveform


def load_audio_features_from_waveform(waveform: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=30,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    if RMS_FEATURE:
        rms = librosa.feature.rms(y=waveform, hop_length=HOP_LENGTH, frame_length=1024).T
        rms_log = np.log1p(rms)
        min_frames = min(mel_db.shape[0], rms_log.shape[0])
        mel_db = mel_db[:min_frames, :]
        rms_log = rms_log[:min_frames, :]
        mel_db = np.concatenate([mel_db, rms_log], axis=1)
    if mel_db.shape[0] > MAX_FRAMES:
        mel_db = mel_db[:MAX_FRAMES, :]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def compute_audio_metrics(waveform: np.ndarray, sr: int) -> dict:
    metrics = {}
    duration = float(len(waveform) / sr) if sr else 0.0
    metrics["duration_s"] = duration
    metrics["sample_rate_hz"] = sr
    if waveform.size:
        metrics["rms"] = float(np.sqrt(np.mean(np.square(waveform))))
        metrics["zero_cross_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(waveform)))
        metrics["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
        metrics["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr)))
        metrics["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr)))
        metrics["peak_amplitude"] = float(np.max(np.abs(waveform)))
    else:
        metrics["rms"] = 0.0
        metrics["zero_cross_rate"] = 0.0
        metrics["spectral_centroid"] = 0.0
        metrics["spectral_bandwidth"] = 0.0
        metrics["spectral_rolloff"] = 0.0
        metrics["peak_amplitude"] = 0.0
    return metrics


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def plot_histogram(waveform: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.hist(waveform, bins=60, color="#2c7fb8", alpha=0.8)
    ax.set_title("Waveform Amplitude Histogram")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    return _fig_to_image(fig)


def plot_attr_overlay(mel: np.ndarray, attr: np.ndarray, title: str) -> np.ndarray:
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.imshow(mel.T, origin="lower", aspect="auto", cmap="viridis")
    ax.imshow(attr.T, origin="lower", aspect="auto", cmap="magma", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    return _fig_to_image(fig)


def plot_waveform_intersection(waveform: np.ndarray, sr: int, frame_attr: np.ndarray, title: str) -> np.ndarray:
    if waveform.size == 0:
        fig, _ = plt.subplots(1, 1, figsize=(6, 2))
        return _fig_to_image(fig)
    frame_times = np.arange(frame_attr.shape[0]) * HOP_LENGTH / sr
    wave_times = np.arange(waveform.shape[0]) / sr
    attr_norm = (frame_attr - frame_attr.min()) / (frame_attr.max() - frame_attr.min() + 1e-6)
    attr_interp = np.interp(wave_times, frame_times, attr_norm)
    max_amp = float(np.max(np.abs(waveform)) + 1e-6)
    attr_scaled = (attr_interp * 2 - 1) * max_amp
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
    ax.plot(wave_times, waveform, color="steelblue", linewidth=0.8, label="Waveform")
    ax.plot(wave_times, attr_scaled, color="darkorange", linewidth=0.8, alpha=0.8, label="Attribution")
    mask = attr_interp > 0.6
    ax.fill_between(wave_times, -max_amp, max_amp, where=mask, color="orange", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    return _fig_to_image(fig)


def _norm_attr(a: np.ndarray) -> np.ndarray:
    a = np.abs(a)
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def _forward_fixed_lengths(model: nn.Module, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return model(x, lengths)


def _target_prob(model: nn.Module, inputs: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(inputs, lengths)
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, labels.view(-1, 1)).squeeze(1)


def _fidelity_curves(model: nn.Module, sample: torch.Tensor, length: torch.Tensor, label: torch.Tensor, attr: torch.Tensor, steps: int = 8):
    flat_attr = attr.abs().flatten()
    order = torch.argsort(flat_attr, descending=True)
    total = flat_attr.numel()
    fractions = torch.linspace(0, 1, steps + 1, device=attr.device)
    del_scores = []
    ins_scores = []
    base = torch.zeros_like(sample)
    flat_sample = sample.flatten()
    for frac in fractions:
        k = int(frac.item() * total)
        idx = order[:k]
        del_sample = flat_sample.clone()
        del_sample[idx] = 0
        del_sample = del_sample.view_as(sample)
        ins_sample = base.flatten()
        ins_sample[idx] = flat_sample[idx]
        ins_sample = ins_sample.view_as(sample)
        del_prob = _target_prob(model, del_sample.unsqueeze(0), length.unsqueeze(0), label.unsqueeze(0)).item()
        ins_prob = _target_prob(model, ins_sample.unsqueeze(0), length.unsqueeze(0), label.unsqueeze(0)).item()
        del_scores.append(del_prob)
        ins_scores.append(ins_prob)
    return fractions.detach().cpu().numpy(), np.array(del_scores), np.array(ins_scores)


def _sensitivity(ig: IntegratedGradients, sample: torch.Tensor, label: torch.Tensor, noise_std: float = 0.02, runs: int = 4) -> float:
    base_attr = ig.attribute(sample.unsqueeze(0), target=label.item(), baselines=0.0).detach()
    base_norm = base_attr.abs().mean().item() + 1e-6
    scores = []
    for _ in range(runs):
        noisy = sample + noise_std * torch.randn_like(sample)
        noisy_attr = ig.attribute(noisy.unsqueeze(0), target=label.item(), baselines=0.0).detach()
        scores.append((noisy_attr - base_attr).abs().mean().item() / base_norm)
    return float(np.mean(scores))


def compute_xai(model: nn.Module, mel_np: np.ndarray, length: int, target_idx: int, device: torch.device):
    if not CAPTUM_AVAILABLE:
        return None, None, None, None, "Captum is not available. Install captum to enable XAI."
    sample = torch.tensor(mel_np, dtype=torch.float32, device=device).unsqueeze(0)
    sample_len = torch.tensor([length], dtype=torch.long, device=device)
    label = torch.tensor(target_idx, dtype=torch.long, device=device)

    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            ig = IntegratedGradients(lambda x: _forward_fixed_lengths(model, x, sample_len))
            attr_ig = ig.attribute(sample, target=label.item(), baselines=torch.zeros_like(sample))
            attr_ig = attr_ig.detach().cpu().numpy()[0]

            occlusion = Occlusion(lambda x: _forward_fixed_lengths(model, x, sample_len))
            attr_occ = occlusion.attribute(
                sample,
                target=label.item(),
                baselines=0.0,
                sliding_window_shapes=(16, 8),
                strides=(8, 4),
            )
            attr_occ = attr_occ.detach().cpu().numpy()[0]
    finally:
        model.train(was_training)

    attr_ig_norm = _norm_attr(attr_ig)
    attr_occ_norm = _norm_attr(attr_occ)
    ig_img = plot_attr_overlay(mel_np, attr_ig_norm, "Integrated Gradients Attribution")
    occ_img = plot_attr_overlay(mel_np, attr_occ_norm, "Occlusion Attribution")
    frame_attr = attr_ig_norm.mean(axis=1)
    waveform_img = None
    fidelity_img = None
    fidelity_text = ""

    try:
        fracs, del_curve, ins_curve = _fidelity_curves(model, sample.squeeze(0), sample_len.squeeze(0), label, torch.tensor(attr_ig, device=device))
        sensitivity = _sensitivity(ig, sample.squeeze(0), label)
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(fracs, del_curve, label="Deletion")
        ax.plot(fracs, ins_curve, label="Insertion")
        ax.set_xlabel("Fraction of top-attributed bins")
        ax.set_ylabel("Target probability")
        ax.legend()
        fidelity_img = _fig_to_image(fig)
        fidelity_text = (
            f"Deletion AUC (lower is better): {np.trapz(del_curve, fracs):.3f}\n"
            f"Insertion AUC (higher is better): {np.trapz(ins_curve, fracs):.3f}\n"
            f"Sensitivity (lower is better): {sensitivity:.3f}"
        )
    except Exception as exc:
        fidelity_text = f"Fidelity metrics unavailable: {exc}"

    return ig_img, occ_img, frame_attr, fidelity_img, fidelity_text


def parse_ravdess_filename(path: Path):
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal, emotion_code, intensity, statement, repetition, actor = parts
    if modality != "03":
        return None
    rav_emotion_map = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised",
    }
    emotion = rav_emotion_map.get(emotion_code)
    if emotion is None:
        return None
    return {"Emotion": emotion, "Dataset": "RAVDESS"}


def parse_cremad_filename(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    crema_emotion_map = {
        "ANG": "Angry",
        "DIS": "Disgust",
        "FEA": "Fearful",
        "HAP": "Happy",
        "NEU": "Neutral",
        "SAD": "Sad",
    }
    emotion = crema_emotion_map.get(parts[2])
    if emotion is None:
        return None
    return {"Emotion": emotion, "Dataset": "CREMA-D"}


def build_dataset_status():
    counts = {label: {"RAVDESS": 0, "CREMA-D": 0, "Feedback": 0} for label in emotion_order}
    total = 0
    rav_total = 0
    crema_total = 0

    if USE_DATASET_DIRS and RAVDESS_DIR.exists():
        for wav_path in RAVDESS_DIR.rglob("*.wav"):
            row = parse_ravdess_filename(wav_path)
            if row:
                counts[row["Emotion"]]["RAVDESS"] += 1
                rav_total += 1
                total += 1

    if USE_DATASET_DIRS and CREMA_DIR.exists():
        for wav_path in CREMA_DIR.glob("*.wav"):
            row = parse_cremad_filename(wav_path)
            if row:
                counts[row["Emotion"]]["CREMA-D"] += 1
                crema_total += 1
                total += 1

    feedback_total = 0
    if FEEDBACK_LOG.exists():
        try:
            with FEEDBACK_LOG.open("r", encoding="utf-8") as handle:
                handle.readline()
                for line in handle:
                    parts = line.strip().split(",")
                    if len(parts) < 4:
                        continue
                    label = parts[3]
                    if label in counts:
                        counts[label]["Feedback"] += 1
                        feedback_total += 1
        except Exception:
            pass

    headers = ["Emotion", "RAVDESS", "CREMA-D", "Feedback", "Total"]
    rows = []
    for label in emotion_order:
        row = counts[label]
        rows.append([label, row["RAVDESS"], row["CREMA-D"], row["Feedback"], row["RAVDESS"] + row["CREMA-D"] + row["Feedback"]])

    if USE_DATASET_DIRS:
        summary_text = (
            f"Total samples: {total + feedback_total} "
            f"(RAVDESS: {rav_total}, CREMA-D: {crema_total}, Feedback: {feedback_total})."
        )
    else:
        summary_text = f"Total samples (feedback only): {feedback_total}."
    write_dataset_status_csv(headers, rows)
    return summary_text, rows, headers


def write_dataset_status_csv(headers, rows):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_STATUS_CSV.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join([str(item) for item in row]) + "\n")


def format_metrics(metrics: dict) -> str:
    return "\n".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()])


def analyze_audio(audio):
    if audio is None:
        return (
            "No audio provided.",
            None,
            None,
            None,
            None,
            None,
            None,
            "",
            None,
            None,
        )

    sr, waveform = audio
    waveform = ensure_mono(np.asarray(waveform, dtype=np.float32))
    waveform = trim_or_pad_waveform(waveform, sr, MAX_RECORD_SECONDS)
    metrics = compute_audio_metrics(waveform, sr)
    hist_img = plot_histogram(waveform)

    waveform_proc = waveform
    sr_proc = sr
    if sr != SAMPLE_RATE:
        waveform_proc = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr_proc = SAMPLE_RATE
    mel_np = load_audio_features_from_waveform(waveform_proc, sr_proc)
    length = mel_np.shape[0]
    inputs = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.long)
    device = next(MODEL.parameters()).device
    inputs = inputs.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = MODEL(inputs, lengths)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_label.get(pred_idx, str(pred_idx))
    pred_conf = float(np.max(probs))

    prob_rows = [[idx_to_label[i], float(probs[i])] for i in range(len(probs))]

    ig_img, occ_img, frame_attr, fidelity_img, fidelity_text = compute_xai(
        MODEL, mel_np, length, pred_idx, device
    )
    waveform_attr_img = None
    if frame_attr is not None:
        waveform_attr_img = plot_waveform_intersection(
            waveform_proc, SAMPLE_RATE, frame_attr, "Waveform + IG Intersection"
        )

    metrics_text = format_metrics(metrics)
    pred_text = f"Prediction: {pred_label} (confidence {pred_conf:.3f})"

    return (
        pred_text,
        metrics_text,
        hist_img,
        prob_rows,
        ig_img,
        occ_img,
        waveform_attr_img,
        fidelity_text,
        fidelity_img,
        audio,
    )


def save_feedback(audio, predicted_label, correct_label, notes):
    if audio is None:
        return "No audio available to save feedback.", *build_dataset_status()
    sr, waveform = audio
    waveform = ensure_mono(np.asarray(waveform, dtype=np.float32))
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = FEEDBACK_DIR / f"feedback_{timestamp}.wav"
    try:
        import soundfile as sf
        sf.write(out_path, waveform, sr)
    except Exception:
        from scipy.io.wavfile import write as wav_write
        wav_write(out_path, sr, (waveform * 32767).astype(np.int16))

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_LOG.exists():
        FEEDBACK_LOG.write_text("timestamp,audio_path,predicted_label,correct_label,notes\n", encoding="utf-8")
    with FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
        safe_notes = (notes or "").replace(",", " ").replace("\n", " ").strip()
        safe_pred = predicted_label or "Unknown"
        safe_correct = correct_label or safe_pred
        handle.write(f"{timestamp},{out_path.name},{safe_pred},{safe_correct},{safe_notes}\n")

    status_text = f"Feedback saved to {out_path.name}."
    summary_text, rows, headers = build_dataset_status()
    return status_text, summary_text, rows, headers


MODEL = load_model()


summary_text, summary_rows, summary_headers = build_dataset_status()

with gr.Blocks() as demo:
    gr.Markdown(f"# Prototype of Speech Recognition\nModel: `{MODEL_PATH}` ({MODEL_STATUS})")
    gr.Markdown(
        "Render mode: set `MODEL_PATH` to your weights file. "
        "Set `USE_DATASET_DIRS=true` only if you upload datasets."
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="Upload or Record (10 seconds max)",
            sources=["upload", "microphone"],
            type="numpy",
        )
        analyze_button = gr.Button("Analyze")

    pred_text = gr.Markdown("Prediction: ")
    metrics_text = gr.Textbox(label="Audio Metrics", lines=8)
    hist_image = gr.Image(label="Histogram", type="numpy")
    prob_table = gr.Dataframe(
        headers=["Label", "Probability"],
        row_count=len(emotion_order),
        col_count=2,
        label="Class Probabilities",
        interactive=False,
    )

    with gr.Row():
        ig_image = gr.Image(label="Integrated Gradients", type="numpy")
        occ_image = gr.Image(label="Occlusion", type="numpy")

    waveform_image = gr.Image(label="Waveform + Attribution", type="numpy")
    fidelity_text = gr.Textbox(label="XAI Metrics", lines=4)
    fidelity_image = gr.Image(label="Fidelity Curves", type="numpy")

    gr.Markdown("## Feedback")
    feedback_label = gr.Dropdown(choices=emotion_order, label="Correct Label")
    feedback_notes = gr.Textbox(label="Notes (optional)", lines=2)
    feedback_button = gr.Button("Submit Feedback")
    feedback_status = gr.Markdown("")

    gr.Markdown("## Dataset Status")
    dataset_summary = gr.Markdown(summary_text)
    dataset_table = gr.Dataframe(
        value=summary_rows,
        headers=summary_headers,
        row_count=len(emotion_order),
        col_count=len(summary_headers),
        label="Samples by Emotion",
        interactive=False,
    )

    audio_state = gr.State()
    pred_state = gr.State("")

    analyze_button.click(
        analyze_audio,
        inputs=[audio_input],
        outputs=[
            pred_text,
            metrics_text,
            hist_image,
            prob_table,
            ig_image,
            occ_image,
            waveform_image,
            fidelity_text,
            fidelity_image,
            audio_state,
        ],
    ).then(
        lambda pred: pred.split("Prediction: ")[-1].split(" (")[0],
        inputs=[pred_text],
        outputs=[pred_state],
    )

    feedback_button.click(
        save_feedback,
        inputs=[audio_state, pred_state, feedback_label, feedback_notes],
        outputs=[feedback_status, dataset_summary, dataset_table],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
