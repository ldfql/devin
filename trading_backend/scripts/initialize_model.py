import xgboost as xgb
import joblib
import numpy as np
import os

def initialize_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Create a simple initial model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=1.5
    )

    # Train with dummy data
    X = np.random.rand(100, 3)  # RSI, MACD, Volatility
    y = np.random.randint(0, 2, 100)  # Binary classification
    model.fit(X, y)

    # Save model
    joblib.dump(model, 'models/market_cycle_model.joblib')
    print('Initial model created and saved.')

if __name__ == '__main__':
    initialize_model()
