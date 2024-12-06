# Cryptocurrency Price Prediction

A deep learning project for cryptocurrency price prediction using LSTM networks.

## Features

- Real-time data collection from multiple sources (CoinGecko, Binance)
- Comprehensive technical indicator calculation
- Deep learning model with LSTM architecture
- Interactive visualizations using Plotly
- Progress tracking and logging
- Configurable parameters via YAML
- Real-time monitoring dashboard
- Automated development environment setup

## Project Structure

```
Crypto_Prediction/
├── setup_dev.sh          # Development environment setup script
├── crypto_prediction_app # Interactive application script
├── data/                 # Data storage
├── models/               # Saved models
├── src/
│   ├── data_collection/  # Data collection scripts
│   ├── preprocessing/    # Data preprocessing
│   ├── visualization/    # Visualization utilities
│   ├── training/         # Model training code
│   ├── monitoring/       # Monitoring dashboard
│   └── utils/            # Utility functions
├── tests/                # Test files
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
├── logs/                 # Log files
├── results/              # Training results
└── visualizations/       # Generated plots
```

## Quick Start

1. Set up the development environment:
    ```bash
    chmod +x setup_dev.sh
    ./setup_dev.sh
    ```

2. Launch the interactive application:
    ```bash
    chmod +x crypto_prediction_app
    ./crypto_prediction_app
    ```

3. Start the monitoring dashboard:
    ```bash
    streamlit run src/monitoring/dashboard.py
    ```

## Detailed Installation

1. Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Interactive Application

The `crypto_prediction_app` script provides an interactive way to execute the pipeline, allowing users to:
1. View the default configuration.
2. Choose to run specific modes (`collect-data`, `train`, `predict`, or `full-pipeline`).
3. Optionally specify a custom configuration file.

Run the script:
```bash
./crypto_prediction_app
```

This interactive application will guide you through:
1. Updating and collecting the latest data.
2. Preprocessing the collected data.
3. Training a new LSTM model.
4. Generating predictions using the trained model.

## Advanced Usage

### Training a New Model
Train the model independently:
```bash
python main.py --config configs/config.yaml --log-level INFO --mode train
```

### Making Predictions Only
Use a pre-trained model to generate predictions:
```bash
python main.py --config configs/config.yaml --log-level INFO --mode predict
```

### Updating Data Only
Update data without proceeding to model training:
```bash
python main.py --config configs/config.yaml --log-level INFO --mode collect-data
```

## Monitoring Dashboard

The monitoring dashboard provides real-time insights into:
- Model performance metrics
- System resource utilization
- Data pipeline health
- Live predictions
- Training history

### Start the Dashboard
1. Ensure Streamlit is installed:
    ```bash
    pip install streamlit
    ```

2. Run the dashboard:
    ```bash
    streamlit run src/monitoring/dashboard.py
    ```

3. Open your browser to `http://localhost:8501`.

## Development Environment

The `setup_dev.sh` script automates setting up your development environment:
- Creates a virtual environment.
- Installs dependencies.
- Sets up the project structure.
- Initializes git hooks.
- Configures Jupyter kernels.
- Sets up environment variables.

## Configuration

The project is configured using:
- `configs/config.yaml`: Main configuration file for paths, model settings, and training options.
- Environment variables: For dynamic configurations.
- Command-line arguments: For runtime flexibility.

## Testing

Run all tests using `pytest`:
```bash
pytest tests/
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

## License

[MIT License](LICENSE)

## Support

For assistance:
1. Check the documentation.
2. Search existing issues.
3. Create a new issue providing:
   - A clear problem description.
   - Steps to reproduce the issue.
   - Error messages and logs.
   - Your system information.
