from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from src.utils.config import Config
from src.training.model import CryptoPredictor
from src.training.trainer import ModelTrainer

if __name__ == "__main__":
    # Load configuration
    config_path = "configs/config.yaml"
    config = Config(config_path)

    # Define paths using configuration
    processed_data_dir = Path(config.config['paths']['processed_data_dir'])
    coin = "bitcoin"  # Specify the coin to load
    data_dir = processed_data_dir / coin

    # Load the real data
    print("Loading data...")
    try:
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    unique_values = np.unique(y_test)
    print("Unique values in y_test:", unique_values)

    # Visualize y_test
    plt.plot(y_test)
    plt.title('y_test Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Close Price')
    plt.show()

    # Initialize the model
    input_shape = X_train.shape[1:]  # Automatically infer input shape
    model_config = config.get_model_config()
    model = CryptoPredictor(
        input_shape=input_shape,
        lstm_units=model_config['lstm_units'],
        dropout_rate=model_config['dropout_rate'],
        dense_units=model_config['dense_units'],
        learning_rate=model_config['learning_rate']
    )
    model.build()
    model.compile()

    # Create the trainer
    training_config = config.get_training_config()
    trainer = ModelTrainer(
        model=model,
        batch_size=training_config['batch_size'],
        epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping_patience'],
        reduce_lr_patience=training_config['reduce_lr_patience'],
        model_dir=config.config['paths']['models_dir']
    )

    # Train the model
    try:
        print("Starting training...")
        history = trainer.train(X_train, y_train, X_val, y_val)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        exit()

    # Evaluate the model
    try:
        print("Evaluating the model...")
        evaluation_results = trainer.evaluate(X_test, y_test)
        print(f"Evaluation Results: {evaluation_results}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        exit()

    # Debug predictions
    try:
        print("Generating predictions...")
        sample_predictions = model.predict(X_test[:5])  # Predict on a small sample for debugging
        print(f"Sample Predictions: {sample_predictions}")
        print(f"Sample Actual Values: {y_test[:5]}")
    except Exception as e:
        print(f"Prediction failed: {e}")
