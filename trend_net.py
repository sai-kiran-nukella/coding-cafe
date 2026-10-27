import numpy as np
import pandas as pd
from simulator import get_simulation_data

# --- Math Utilities ---

def softmax(x):
    """
    Computes the softmax activation for multiclass classification.
    Subtracts max(x) for numerical stability (prevents overflow).
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def tanh(x):
    """Hyperbolic tangent activation (ranges -1 to 1). Good for sequence data."""
    return np.tanh(x)

def tanh_deriv(x):
    """Derivative of tanh: 1 - tanh(x)^2"""
    return 1.0 - np.power(np.tanh(x), 2)

def to_one_hot(y, num_classes=3):
    """Converts integer labels (0, 1, 2) to one-hot vectors."""
    return np.eye(num_classes)[y]

# --- The Neural Network ---

class TrendNet:
    def __init__(self, input_size, hidden_size=16, output_size=3, learning_rate=0.01):
        self.lr = learning_rate
        
        # Xavier/Glorot Initialization
        # Weights initialized with variance 1/n
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Cache
        self.z1, self.a1, self.z2 = None, None, None

    def forward(self, X):
        """
        Forward Pass:
        Input -> Linear -> Tanh -> Linear -> Softmax
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return softmax(self.z2)

    def backward(self, X, y_one_hot, probs):
        """
        Backpropagation using Cross-Entropy Loss Gradient.
        """
        # FIX 1: Extract ONLY the number of rows (batch size)
        m = X.shape[0] 
        
        # 1. Gradient of Cross-Entropy Loss w.r.t Softmax Input (z2)
        # dL/dz2 = prob - y_true
        # Division by m normalizes the gradient by batch size
        grad_z2 = (probs - y_one_hot) / m
        
        # 2. Output Layer Gradients
        d_W2 = np.dot(self.a1.T, grad_z2)
        d_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        
        # 3. Hidden Layer Gradients
        # Backprop through Tanh: dL/dz1 = (dL/dz2 * W2.T) * tanh'(z1)
        d_hidden = np.dot(grad_z2, self.W2.T) * tanh_deriv(self.z1)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # 4. Updates
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

    def train(self, X, y, epochs=2000):
        y_oh = to_one_hot(y, 3)
        for _ in range(epochs):
            probs = self.forward(X)
            self.backward(X, y_oh, probs)

# --- Data Preparation & Global State ---

# Configuration
WINDOW_SIZE = 3 # Look at last 3 months
COLS = ['market_price', 'cost_factor', 'demand_factor'] 

# Global stores for inference
_model = None
_x_mean = 0
_x_std = 1

def create_sequences(df):
    """
    Transforms the linear dataframe into a supervised learning dataset.
    Input (X): Flattened array of last 3 months data.
    Output (y): Direction of price movement in the 4th month.
    """
    # FIX 2: Correct list initialization syntax
    X, y = [], []
    
    for good in df['good'].unique():
        raw_data = df[df['good'] == good][COLS].values
        
        for i in range(len(raw_data) - WINDOW_SIZE - 1):
            # Input: Flatten 3 months of 3 features
            seq = raw_data[i : i + WINDOW_SIZE].flatten()
            
            # Target: Percentage change 1 month ahead
            curr = raw_data[i + WINDOW_SIZE - 1, 0]
            fut = raw_data[i + WINDOW_SIZE, 0]
            pct = (fut - curr) / curr
            
            # Labeling: 0=Fall, 1=Flat, 2=Rise
            if pct < -0.02: label = 0
            elif pct > 0.02: label = 2
            else: label = 1
                
            X.append(seq)
            y.append(label)
            
    return np.array(X), np.array(y)

def train_trend_model():
    global _model, _x_mean, _x_std
    
    # 1. Get Data
    df = get_simulation_data()
    X, y = create_sequences(df)
    
    # 2. Normalize
    _x_mean = np.mean(X, axis=0)
    _x_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - _x_mean) / _x_std
    
    # 3. Train
    # FIX 3: Pass ONLY the number of features (columns) as input_size
    input_dim = X.shape[1]
    
    _model = TrendNet(input_size=input_dim, hidden_size=16, output_size=3)
    _model.train(X_scaled, y, epochs=3000)
    
    # 4. Validation
    probs = _model.forward(X_scaled)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)
    
    print(f"\n--- Trend Net Validation ---")
    print(f"Training Accuracy: {acc*100:.1f}%")

def predict_trend(recent_history_df):
    if len(recent_history_df) != WINDOW_SIZE:
        return {"direction": "Flat", "confidence": 50.0, "magnitude_percent": 0.0}
        
    raw_vector = recent_history_df[COLS].values.flatten()
    input_scaled = (raw_vector - _x_mean) / _x_std
    input_scaled = input_scaled.reshape(1, -1)
    
    probs = _model.forward(input_scaled)
    pred_idx = np.argmax(probs)
    confidence = probs[0, pred_idx]
    
    labels = {0: "Fall", 1: "Flat", 2: "Rise"}
    direction = labels[pred_idx]
    
    base_mag = 0.5 if direction == "Flat" else 2.0
    magnitude = base_mag + (confidence * 2.0)
    
    return {
        "direction": direction,
        "confidence": round(confidence * 100, 1),
        "magnitude_percent": round(magnitude, 1)
    }

if __name__ == "__main__":
    train_trend_model()
