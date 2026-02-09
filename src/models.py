import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, GRU, Dense, Dropout, Concatenate, TimeDistributed
from tensorflow.keras.optimizers import Adam
from .layers import Attention, VariableSelectionNetwork, GatingLayer, GRN

def build_bilstm_model(input_shape, learning_rate=0.001):
    """
    Builds a Bidirectional LSTM model for baseline comparison.
    """
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Bi-LSTM")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    return model

def build_gru_attention_model(input_shape, learning_rate=0.001):
    """
    Builds a GRU model with a custom Attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    
    # Custom Attention Layer
    context_vector = Attention()(x)
    
    x = Dense(32, activation='relu')(context_vector)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="GRU_Attention")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    return model

def build_tft_model(input_shape, hidden_dim=64, dropout_rate=0.1, learning_rate=0.001):
    """
    Builds the Temporal Fusion Transformer (TFT) adapted for microbiome data.
    Features:
    - Variable Selection Network (VSN) for feature importance
    - Gated Residual Network (GRN) for non-linear processing
    - Gating Layer for physiological gating
    """
    inputs = Input(shape=input_shape)
    
    # 1. Variable Selection Network
    vsn_output, weights = VariableSelectionNetwork(hidden_dim=hidden_dim, 
                                                   dropout_rate=dropout_rate)(inputs)
    
    # 2. LSTM Encoder (Temporal Processing)
    lstm_out = LSTM(hidden_dim, return_sequences=True)(vsn_output)
    
    # 3. Gated Residual Network (GRN) - Applied per time step
    # We use TimeDistributed to apply the GRN to each time step independently
    grn_out = TimeDistributed(GRN(hidden_dim=hidden_dim, dropout_rate=dropout_rate))(lstm_out)
    
    # 4. Gating Layer (Physiological Gating)
    # Serves as the "Sentinel" to filter noise
    gated_out = TimeDistributed(GatingLayer(hidden_dim=hidden_dim))(grn_out)
    
    # 5. Global Attention
    context_vector = Attention()(gated_out)
    
    # 6. Final Classification Head
    outputs = Dense(1, activation='sigmoid')(context_vector)
    
    model = Model(inputs=inputs, outputs=outputs, name="Adapted_TFT")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    
    return model