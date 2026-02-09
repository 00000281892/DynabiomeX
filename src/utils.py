# src/utils.py

import os
import tensorflow as tf
from .layers import Attention, GRN, VariableSelectionNetwork, GatingLayer


def load_trained_model(model_key, base_path='../results/saved_models'):
    """
    Loads one of the 3 specific pre-trained DynaBiomeX models.

    Args:
        model_key (str): One of ['tft', 'bilstm', 'gru'].
        base_path (str): Path to the folder containing .keras files.

    Returns:
        tf.keras.Model: The loaded model.
    """
    # Map short keys to your exact filenames
    filename_map = {
        'tft': 'best_enhanced_tft_model.keras',
        'bilstm': 'best_bilstm_model_tuned.keras',
        'gru': 'best_gru_attention_tuned_II.keras'
    }

    if model_key not in filename_map:
        raise ValueError(
            f"❌ Unknown model key: {model_key}. Choose from {list(filename_map.keys())}")

    full_path = os.path.join(base_path, filename_map[model_key])

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Model file not found at: {full_path}")

    # Custom objects dictionary for Keras loading
    custom_objects = {
        'Attention': Attention,
        'GRN': GRN,
        'VariableSelectionNetwork': VariableSelectionNetwork,
        'GatingLayer': GatingLayer
    }

    try:
        # 'compile=False' is often safer for inference if you don't need to resume training
        model = tf.keras.models.load_model(
            full_path, custom_objects=custom_objects, compile=False)
        print(
            f"✅ Successfully loaded {model_key.upper()} from {filename_map[model_key]}")
        return model
    except Exception as e:
        print(f"❌ Error loading {model_key}: {e}")
        return None
