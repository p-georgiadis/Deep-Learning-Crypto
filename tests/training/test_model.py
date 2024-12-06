import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential


@pytest.fixture
def sample_data():
    """Create sample data for model testing."""
    X = np.random.random((100, 60, 20))  # 100 samples, 60 timesteps, 20 features
    y = np.random.random((100, 1))  # 100 samples, 1 target value
    return X, y


@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        'input_shape': (60, 20),
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'dense_units': [16],
        'learning_rate': 0.001
    }


def test_model_initialization(model_config):
    """Test model initialization."""
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()  # Explicitly build the model

    assert isinstance(model.model, Sequential)
    assert model.input_shape == model_config['input_shape']
    assert model.lstm_units == model_config['lstm_units']
    assert model.dropout_rate == model_config['dropout_rate']
    assert model.dense_units == model_config['dense_units']



def test_model_build(model_config):
    """Test model building."""
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()

    # Validate model structure
    assert len(model.model.layers) > 0
    assert model.model.input_shape == (None, *model_config['input_shape'])
    assert model.model.output_shape == (None, 1)  # Single-step prediction


def test_model_compile(model_config):
    """Test model compilation."""
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()

    assert model.model.optimizer is not None
    assert model.model.loss == 'mse'
    assert len(model.model.metrics) > 0


def test_model_fit(sample_data, model_config):
    """Test model fitting."""
    from src.training.model import CryptoPredictor

    X, y = sample_data
    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()

    # Split data for validation
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # Use explicit validation data
        epochs=2,
        batch_size=32
    )

    assert isinstance(history, tf.keras.callbacks.History)
    assert len(history.history['loss']) == 2
    assert 'val_loss' in history.history



def test_model_predict(sample_data, model_config):
    """Test model prediction."""
    from src.training.model import CryptoPredictor

    X, _ = sample_data
    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()

    predictions = model.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X), 1)
    assert not np.any(np.isnan(predictions))


def test_model_save_load(tmp_path, model_config, sample_data):
    """Test model saving and loading."""
    from src.training.model import CryptoPredictor

    X, y = sample_data
    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()
    model.fit(X, y, epochs=1)

    # Save model
    save_path = tmp_path / "test_model.keras"  # Use `.keras` extension
    model.save(save_path)

    # Load model
    loaded_model = CryptoPredictor.load(save_path)

    # Compare predictions
    original_pred = model.predict(X)
    loaded_pred = loaded_model.predict(X)
    np.testing.assert_array_almost_equal(original_pred, loaded_pred)



def test_invalid_model_configuration():
    """Test model initialization with invalid configurations."""
    from src.training.model import CryptoPredictor

    with pytest.raises(ValueError):
        CryptoPredictor(input_shape=(0, 20))  # Invalid input shape
    with pytest.raises(ValueError):
        CryptoPredictor(input_shape=(60, 20), lstm_units=[-32])  # Invalid LSTM units
    with pytest.raises(ValueError):
        CryptoPredictor(input_shape=(60, 20), dropout_rate=1.5)  # Invalid dropout rate


def test_custom_loss_functions():
    """Test custom loss functions."""
    from src.training.model import directional_accuracy

    y_true = tf.constant([[1.0], [2.0], [1.5], [3.0]])
    y_pred = tf.constant([[1.2], [1.8], [1.6], [2.8]])

    accuracy = directional_accuracy(y_true, y_pred)
    assert isinstance(accuracy, tf.Tensor)
    assert 0 <= float(accuracy) <= 1


def test_model_evaluation(sample_data, model_config):
    """Test model evaluation."""
    from src.training.model import CryptoPredictor

    X, y = sample_data
    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()
    model.fit(X, y, epochs=1)

    evaluation = model.evaluate(X, y)
    assert isinstance(evaluation, list)
    assert len(evaluation) > 0  # Includes loss and metrics


def test_model_summary(model_config):
    """Test model summary generation."""
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()

    summary = []
    model.model.summary(print_fn=lambda x: summary.append(x))

    assert len(summary) > 0
    assert any('lstm' in line.lower() for line in summary)
    assert any('dense' in line.lower() for line in summary)
