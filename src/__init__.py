# src/__init__.py
from .utils import load_trained_model
# 1. Expose Custom Layers
from .layers import (
    Attention,
    GRN,
    VariableSelectionNetwork,
    GatingLayer
)

# 2. Expose Model Builders
from .models import (
    build_bilstm_model,
    build_gru_attention_model,
    build_tft_model
)

# 3. Expose Evaluation Metrics & Plotting
from .evaluation import (
    calculate_calibration_metrics,
    plot_calibration_curve
)

# Gets imported when someone uses "from src import *"
__all__ = [
    "Attention",
    "GRN",
    "VariableSelectionNetwork",
    "GatingLayer",
    "build_bilstm_model",
    "build_gru_attention_model",
    "build_tft_model",
    "calculate_calibration_metrics",
    "plot_calibration_curve"
]
