"""
Temporal swelling predictor using a 3D CNN.

This module takes a sequence of frames (a short video clip) and predicts:
  1. Swelling behaviour class: 'rapid', 'moderate', 'slow', 'invalid'
  2. Predicted bed height at the next N time steps (regression)

Architecture:
  - Pretrained R3D-18 backbone (ResNet-based 3D CNN, trained on Kinetics-400)
  - Custom classification and regression heads
  - Input: (batch, channels=3, frames=16, height=112, width=112)

Integration:
  - Called from height_inference.py after YOLO analysis is complete
  - Takes the saved frame images + CSV output as inputs
  - Outputs are appended to the results and saved as additional plots/CSV columns
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from src.logger import setup_logger

logger = setup_logger("temporal_predictor")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CLIP_FRAMES    = 16          # Number of consecutive frames per clip
FRAME_SIZE     = (112, 112)  # Spatial resize for R3D-18 input
PREDICT_STEPS  = 5           # How many future time steps to predict (regression)

# Swelling behaviour classes
SWELLING_CLASSES = ['rapid', 'moderate', 'slow', 'invalid']

# ImageNet normalisation (used by pretrained R3D-18)
MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.22145,  0.216989]


# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────

class SwellingPredictor3D(nn.Module):
    """
    3D CNN for temporal swelling analysis.

    Two output heads:
      - classifier:  predicts swelling behaviour class (4-way softmax)
      - regressor:   predicts next PREDICT_STEPS bed height values (mm)
    """

    def __init__(self, num_classes: int = 4, predict_steps: int = PREDICT_STEPS):
        super().__init__()

        # Load pretrained R3D-18 backbone
        backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)

        # Remove the original fully-connected layer, keep the feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # backbone outputs (batch, 512, 1, 1, 1) → flatten to (batch, 512)

        feature_dim = 512

        # Classification head: rapid / moderate / slow / invalid
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Regression head: predict next N bed heights
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, predict_steps)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, 3, frames, H, W)

        Returns:
            class_logits:  (batch, num_classes)
            height_preds:  (batch, predict_steps)  — predicted future heights in mm
        """
        features = self.backbone(x)          # (batch, 512, 1, 1, 1)
        class_logits  = self.classifier(features)
        height_preds  = self.regressor(features)
        return class_logits, height_preds


# ─────────────────────────────────────────────
# Frame Preprocessing
# ─────────────────────────────────────────────

def preprocess_frames(frame_paths: List[str]) -> torch.Tensor:
    """
    Load and preprocess a list of image paths into a model-ready tensor.

    Args:
        frame_paths: List of exactly CLIP_FRAMES image file paths

    Returns:
        Tensor of shape (1, 3, CLIP_FRAMES, 112, 112) ready for inference
    """
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, FRAME_SIZE)
        img = img.astype(np.float32) / 255.0

        # Normalise with ImageNet stats
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - MEAN[c]) / STD[c]

        frames.append(img)

    # Stack to (frames, H, W, C) then rearrange to (C, frames, H, W)
    clip = np.stack(frames, axis=0)                    # (16, 112, 112, 3)
    clip = np.transpose(clip, (3, 0, 1, 2))            # (3, 16, 112, 112)
    tensor = torch.tensor(clip, dtype=torch.float32)
    return tensor.unsqueeze(0)                         # (1, 3, 16, 112, 112)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def load_predictor(model_path: str, device: str = None) -> Tuple[SwellingPredictor3D, str]:
    """
    Load a trained SwellingPredictor3D from disk.

    Args:
        model_path: Path to saved .pt weights file
        device:     'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Returns:
        (model, device_str)
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'        # Apple Silicon
        else:
            device = 'cpu'

    model = SwellingPredictor3D()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(f"Loaded 3D CNN from {model_path} on {device}")
    return model, device


def run_temporal_prediction(
    model: SwellingPredictor3D,
    frame_paths: List[str],
    device: str,
    known_heights_mm: List[float]
) -> Dict:
    """
    Run 3D CNN inference over a full experiment's frames using a sliding window.

    Args:
        model:             Loaded SwellingPredictor3D
        frame_paths:       All frame image paths in chronological order
        device:            Device string
        known_heights_mm:  Bed heights (mm) from YOLO analysis, used to label clips

    Returns:
        Dictionary with:
          - 'clip_times':       list of clip centre timestamps (frame indices)
          - 'swelling_classes': list of predicted class strings per clip
          - 'swelling_probs':   list of class probability dicts per clip
          - 'future_heights':   list of arrays of predicted future heights per clip
    """
    n_frames = len(frame_paths)
    if n_frames < CLIP_FRAMES:
        logger.warning(
            f"Only {n_frames} frames available, need {CLIP_FRAMES} minimum for 3D CNN. "
            f"Skipping temporal prediction."
        )
        return {}

    clip_times      = []
    swelling_classes = []
    swelling_probs   = []
    future_heights   = []

    # Stride of CLIP_FRAMES//2 so clips overlap (more predictions, smoother output)
    stride = CLIP_FRAMES // 2

    with torch.no_grad():
        for start in range(0, n_frames - CLIP_FRAMES + 1, stride):
            end   = start + CLIP_FRAMES
            clip  = frame_paths[start:end]
            mid   = start + CLIP_FRAMES // 2     # Centre frame index

            try:
                tensor = preprocess_frames(clip).to(device)
                class_logits, height_preds = model(tensor)

                # Classification
                probs      = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                class_idx  = int(probs.argmax())
                class_name = SWELLING_CLASSES[class_idx]
                prob_dict  = {cls: float(p) for cls, p in zip(SWELLING_CLASSES, probs)}

                # Regression
                future_mm = height_preds[0].cpu().numpy().tolist()

                clip_times.append(mid)
                swelling_classes.append(class_name)
                swelling_probs.append(prob_dict)
                future_heights.append(future_mm)

                logger.info(
                    f"Clip centred on frame {mid}: "
                    f"class={class_name} ({probs.max():.2f}), "
                    f"next heights={[f'{v:.1f}' for v in future_mm[:3]]}..."
                )

            except Exception as e:
                logger.warning(f"Clip starting frame {start} failed: {e}")
                continue

    return {
        'clip_times':       clip_times,
        'swelling_classes': swelling_classes,
        'swelling_probs':   swelling_probs,
        'future_heights':   future_heights
    }


# ─────────────────────────────────────────────
# Saving Results
# ─────────────────────────────────────────────

def save_temporal_results(
    temporal_results: Dict,
    output_dir: str,
    frame_interval_s: float = 2.0
) -> Tuple[str, str]:
    """
    Save temporal prediction results as a CSV and a plot.

    Args:
        temporal_results:  Output from run_temporal_prediction()
        output_dir:        Directory to write files into
        frame_interval_s:  Seconds between frames (used to convert frame idx → seconds)

    Returns:
        (csv_path, plot_path)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    clip_times      = temporal_results['clip_times']
    swelling_classes = temporal_results['swelling_classes']
    swelling_probs   = temporal_results['swelling_probs']
    future_heights   = temporal_results['future_heights']

    # Convert frame indices to seconds
    times_s = [t * frame_interval_s for t in clip_times]

    # ── CSV ──────────────────────────────────────────────────
    rows = []
    for t, cls, probs, future in zip(times_s, swelling_classes, swelling_probs, future_heights):
        row = {'time_s': t, 'predicted_class': cls}
        row.update({f'prob_{k}': v for k, v in probs.items()})
        row.update({f'future_h_{i+1}': v for i, v in enumerate(future)})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "temporal_prediction.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved temporal CSV: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────
    colour_map = {
        'rapid':    '#e74c3c',
        'moderate': '#f39c12',
        'slow':     '#3498db',
        'invalid':  '#2ecc71'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Classification probability over time
    for cls in SWELLING_CLASSES:
        probs_series = [p[cls] for p in swelling_probs]
        ax1.plot(times_s, probs_series, label=cls.title(),
                 color=colour_map[cls], linewidth=2)

    ax1.set_ylabel("Class Probability", fontsize=12)
    ax1.set_title("3D CNN Swelling Behaviour Classification", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Shade background by predicted class
    for i in range(len(times_s) - 1):
        ax1.axvspan(times_s[i], times_s[i+1],
                    alpha=0.08, color=colour_map[swelling_classes[i]])

    # Panel 2: Predicted future heights (first prediction step only for clarity)
    future_step1 = [f[0] if f else None for f in future_heights]
    future_step1 = [v for v in future_step1 if v is not None]
    ax2.plot(times_s[:len(future_step1)], future_step1,
             's--', color='purple', linewidth=2, markersize=5,
             label='Predicted height (next step)')
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Predicted Bed Height (mm)", fontsize=12)
    ax2.set_title("3D CNN Bed Height Regression (Next Time Step)", fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "temporal_prediction.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved temporal plot: {plot_path}")

    return csv_path, plot_path
