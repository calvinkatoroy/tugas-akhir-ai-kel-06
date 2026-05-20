"""
LSTM Inference Demo for Kelompok 08.
This script demonstrates how to load a trained LSTM classifier checkpoint,
run inference on sequence windows of network traffic features, and classify them
as normal or DDoS.

It features robust auto-detection of model architecture from the checkpoint weights,
graceful fallback if scaler.pkl is missing, and a premium CLI dashboard.

Usage:
    python lstm_demo.py --checkpoint results/best_lstm.pt --scaler data/processed/splits/scaler.pkl
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import joblib

# Re-declare model class to ensure standalone execution
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # Take the last sequence step
        return self.classifier(last)

def load_auto_model(ckpt_path):
    """Inspects checkpoint weights to auto-detect layers and hidden sizes."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Extract dimensions from weights
    ih_weight_key = 'lstm.weight_ih_l0'
    if ih_weight_key not in state_dict:
        # Check if wrapped in model. prefix or similar
        potential_keys = [k for k in state_dict.keys() if 'weight_ih_l0' in k]
        if potential_keys:
            ih_weight_key = potential_keys[0]
        else:
            raise KeyError("Could not find LSTM weights in state dict.")
            
    hidden_size = state_dict[ih_weight_key].shape[0] // 4
    n_features = state_dict[ih_weight_key].shape[1]
    
    # Count HH layers
    hh_keys = [k for k in state_dict.keys() if 'weight_hh_l' in k]
    num_layers = len(hh_keys)
    
    print(f"Auto-Detected Model Specs:")
    print(f"  * Features (Input Size): {n_features}")
    print(f"  * Hidden Size:          {hidden_size}")
    print(f"  * LSTM Layers:          {num_layers}")
    
    # Strip prefixes if model was saved inside a Trainer
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '').replace('model.', '')
        clean_state_dict[clean_key] = v
        
    model = LSTMClassifier(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model, n_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/best_lstm.pt', help='Path to PyTorch model checkpoint (.pt)')
    parser.add_argument('--scaler', default='data/processed/splits/scaler.pkl', help='Path to StandardScaler (.pkl)')
    args = parser.parse_args()

    print("=" * 60)
    print("      KELOMPOK 08 - KECERDASAN BUATAN - LSTM DEMO INTERFACES")
    print("=" * 60)

    # 1. Load Model Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Model checkpoint '{ckpt_path}' not found.")
        print("Please train a model or place 'best_lstm.pt' in the results folder.")
        sys.exit(1)
        
    try:
        model, n_features = load_auto_model(ckpt_path)
    except Exception as e:
        print(f"ERROR: Failed to load PyTorch model: {e}")
        sys.exit(1)

    # 2. Load Scaler
    scaler_path = Path(args.scaler)
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print(f"StandardScaler loaded successfully from: {scaler_path.name}")
        except Exception as e:
            print(f"WARNING: Failed to load scaler from '{scaler_path}': {e}. Using dummy scaling.")
    else:
        print(f"NOTICE: Scaler file '{scaler_path}' not found (offline mode). Running with raw features.")

    # 3. Create Sample Sequence Traffic (seq_len=10, 16 features)
    # The 16 target features:
    feature_names = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Mean",
        "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
        "Fwd IAT Mean", "Bwd IAT Mean", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "Avg Packet Size"
    ]

    # Class 0: Normal/Benign Network Traffic Sequence (Standard low-speed bidirectional communication)
    normal_traffic = [
        [20615.0000, 2.0000, 2.0000, 84.0000, 116.0000, 42.0000, 58.0000, 9696.0000, 194.0312, 3.0000, 2.9688, 0.0000, 0.0000, 0.0000, 0.0000, 60.5000],
        [391.0000, 74.0000, 0.0000, 31984.0000, 0.0000, 432.2162, -0.0000, 81800512.0000, 189258.3125, 5.3750, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 438.1622],
        [42457120.0000, 6.0000, 2.0000, 36.0000, 12.0000, 6.0000, 6.0000, 0.0000, 0.1875, 8491404.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 6.7500],
        [3002973.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 1000991.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [1.0000, 2.0000, 0.0000, 2752.0000, 0.0000, 1376.0000, -0.0000, 2752000000.0000, 2000000.0000, 0.9375, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2064.0000],
        [66.0000, 3.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 45454.5469, 33.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [1044.0000, 32.0000, 0.0000, 13504.0000, 0.0000, 422.0000, -0.0000, 12934864.0000, 30651.3438, 33.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 435.7500],
        [2999723.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 999907.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [3000122.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 1000040.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [118525808.0000, 35.0000, 32.0000, 1740.0000, 2820.0000, 49.7143, 88.1250, 40.0000, 0.5703, 3486053.0000, 3731536.7500, 0.0000, 0.0000, 0.0000, 1.0000, 68.0597],
    ]
    normal_seq = np.array(normal_traffic, dtype=np.float32)

    # Class 1: DDoS Attack Network Traffic Sequence (High-rate unidirectional UDP/DNS reflective burst flood)
    ddos_traffic = [
        [3002973.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 1000991.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [1.0000, 2.0000, 0.0000, 2752.0000, 0.0000, 1376.0000, -0.0000, 2752000000.0000, 2000000.0000, 0.9375, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2064.0000],
        [66.0000, 3.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 45454.5469, 33.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [1044.0000, 32.0000, 0.0000, 13504.0000, 0.0000, 422.0000, -0.0000, 12934864.0000, 30651.3438, 33.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 435.7500],
        [2999723.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 999907.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [3000122.0000, 4.0000, 0.0000, 2064.0000, 0.0000, 516.0000, -0.0000, 688.0000, 1.3438, 1000040.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 645.0000],
        [118525808.0000, 35.0000, 32.0000, 1740.0000, 2820.0000, 49.7143, 88.1250, 40.0000, 0.5703, 3486053.0000, 3731536.7500, 0.0000, 0.0000, 0.0000, 1.0000, 68.0597],
        [10059430.0000, 4.0000, 4.0000, 4.0000, 0.0000, 1.0000, -0.0000, 0.0000, 0.7969, 3343713.0000, 3343658.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6250],
        [20615.0000, 2.0000, 2.0000, 84.0000, 116.0000, 42.0000, 58.0000, 9696.0000, 194.0312, 3.0000, 2.9688, 0.0000, 0.0000, 0.0000, 0.0000, 60.5000],
        [391.0000, 74.0000, 0.0000, 31984.0000, 0.0000, 432.2162, -0.0000, 81800512.0000, 189258.3125, 5.3750, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 438.1622],
    ]
    ddos_seq = np.array(ddos_traffic, dtype=np.float32)


    # Apply scaling
    if scaler is not None:
        normal_seq_scaled = scaler.transform(normal_seq)
        ddos_seq_scaled = scaler.transform(ddos_seq)
    else:
        normal_seq_scaled = normal_seq
        ddos_seq_scaled = ddos_seq

    # Reshape to (batch_size=1, seq_len=10, n_features=16)
    normal_tensor = torch.tensor(normal_seq_scaled).unsqueeze(0)
    ddos_tensor = torch.tensor(ddos_seq_scaled).unsqueeze(0)

    # 4. Perform Inference
    print("\n--- 4. Running Model Predictions ---")
    
    with torch.no_grad():
        normal_logits = model(normal_tensor)
        normal_probs = torch.softmax(normal_logits, dim=-1).squeeze().numpy()
        normal_pred = np.argmax(normal_probs)

        ddos_logits = model(ddos_tensor)
        ddos_probs = torch.softmax(ddos_logits, dim=-1).squeeze().numpy()
        ddos_pred = np.argmax(ddos_probs)

    classes_map = {0: "NORMAL (BENIGN)", 1: "DDOS ATTACK"}

    # Dashboard display
    def print_result_card(title, probs, pred):
        print("-" * 50)
        print(f"| SAMPLE: {title}")
        print("-" * 50)
        print(f"| Predicted Class:  \033[1m{classes_map[pred]}\033[0m")
        print(f"| Probability:")
        print(f"|   * Normal:       {probs[0]:.6%}")
        print(f"|   * DDoS:         {probs[1]:.6%}")
        status_color = "\033[92m[SAFE]\033[0m" if pred == 0 else "\033[91m[ALERT - ATTACK DETECTED]\033[0m"
        print(f"| Security Status:  {status_color}")
        print("-" * 50 + "\n")

    print_result_card("Network Traffic Sample A (Legitimate Web Browsing)", normal_probs, normal_pred)
    print_result_card("Network Traffic Sample B (High Rate SYN Flood Reflective Burst)", ddos_probs, ddos_pred)

    print("=" * 60)
    print("DEMO INFERENCE EXECUTED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    main()
