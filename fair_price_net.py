import numpy as np
import pandas as pd
from simulator import get_simulation_data

# --- Math Utilities ---

class Scaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Essential for Neural Network convergence.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        # Add epsilon to prevent division by zero
        self.std = np.std(data, axis=0) + 1e-8
    
    def transform(self, data):
        if self.mean is None:
            raise Exception("Scaler has not been fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def relu(x):
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)

def relu_deriv(x):
    """Derivative of ReLU for backpropagation."""
    return (x > 0).astype(float)

# --- The Neural Network ---

class FairPriceNet:
    def __init__(self, input_size=4, hidden_size=10, learning_rate=0.01):
        self.lr = learning_rate
        
        # He Initialization (Best for ReLU activations)
        # Weights are initialized with variance 2/n to maintain signal variance
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Output Layer (Linear activation for Regression)
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, 1))
        
        # Cache for backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None

    def forward(self, X):
        """Forward pass: Inputs -> Hidden (ReLU) -> Output (Linear)"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        """
        Backpropagation: Calculates gradients and updates weights using Gradient Descent.
        Loss Function: Mean Squared Error (MSE)
        """
        m = y.shape[0] # Batch size
        
        # 1. Gradient of Loss w.r.t Output (MSE derivative)
        # d(Loss)/d(Output) = 2 * (Output - y) / m
        grad_output = 2 * (output - y) / m
        
        # 2. Layer 2 Gradients (Output Layer)
        d_W2 = np.dot(self.a1.T, grad_output)
        d_b2 = np.sum(grad_output, axis=0, keepdims=True)
        
        # 3. Layer 1 Gradients (Hidden Layer)
        # Propagate error back through ReLU derivative
        d_hidden = np.dot(grad_output, self.W2.T) * relu_deriv(self.z1)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # 4. Weight Updates (Stochastic Gradient Descent)
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

    def train(self, X, y, epochs=2000):
        """Training loop."""
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# --- Global State Management ---
# We keep the trained model and scalers in memory for the session
_model = FairPriceNet()
_x_scaler = Scaler()
_y_scaler = Scaler()

def train_and_validate():
    """
    Trains the network and prints a validation report comparing it 
    to a standard Linear Regression baseline.
    """
    # 1. Get Training Data
    df = get_simulation_data()
    
    # Feature Engineering
    # Input: Cost, Demand, Supply, Seasonality
    X = df[['cost_factor', 'demand_factor', 'supply_factor', 'seasonality']].values
    # Target: The True Equilibrium Price (Hidden from the market noise)
    y = df[['true_equilibrium']].values
    
    # 2. Split Data (80% Train, 20% Validate)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 3. Scale Data (Critical for NN performance)
    _x_scaler.fit(X_train)
    _y_scaler.fit(y_train)
    
    X_train_scaled = _x_scaler.transform(X_train)
    y_train_scaled = _y_scaler.transform(y_train)
    X_val_scaled = _x_scaler.transform(X_val)
    
    # 4. Train Neural Network
    # Note: We re-initialize the model to ensure a fresh start on reload
    global _model
    _model = FairPriceNet(hidden_size=12, learning_rate=0.01)
    _model.train(X_train_scaled, y_train_scaled, epochs=2500)
    
    # 5. Evaluate Neural Network
    preds_scaled = _model.forward(X_val_scaled)
    preds_nn = _y_scaler.inverse_transform(preds_scaled)
    mse_nn = np.mean((preds_nn - y_val)**2)
    
    # 6. Evaluate Baseline (Linear Regression)
    # We solve Normal Equation: w = (X^T X)^-1 X^T y
    X_train_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))]) # Add bias term
    X_val_aug = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
    # Use lstsq for stable solution
    weights, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    preds_lin = X_val_aug @ weights
    mse_lin = np.mean((preds_lin - y_val)**2)
    
    # 7. Report Results to Console
    print(f"\n--- Fair Price Net Validation ---")
    print(f"Linear Baseline MSE: {mse_lin:.2f}")
    print(f"Neural Net MSE:      {mse_nn:.2f}")
    if mse_nn < mse_lin:
        print(f"SUCCESS: Neural Net is {mse_lin/mse_nn:.1f}x more accurate than Linear Regression.")
    else:
        print("WARNING: Neural Net underperformed. Check hyperparameters.")

def predict_fair_price(cost, demand, supply, seasonality):
    """
    Inference function for the live app.
    Takes raw economic factors, scales them, runs the net, and returns unscaled price.
    """
    # 1. Structure Input
    X = np.array([[cost, demand, supply, seasonality]])
    
    # 2. Scale Input
    X_scaled = _x_scaler.transform(X)
    
    # 3. Predict
    pred_scaled = _model.forward(X_scaled)
    
    # 4. Inverse Scale Output
    fair_price = _y_scaler.inverse_transform(pred_scaled)
    
    return float(fair_price[0, 0])

if __name__ == "__main__":
    # Allow running this file standalone for debugging
    train_and_validate()
