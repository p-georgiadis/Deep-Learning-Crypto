# main.py
import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_collection.data_collector import DataCollector
from src.preprocessing.pipeline import Pipeline
from src.training.model import CryptoPredictor
from src.training.trainer import ModelTrainer
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.visualizer import CryptoVisualizer

class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for a training progress bar."""

    def __init__(self, epochs):
        super().__init__()
        self.progress_bar = tqdm(total=epochs, desc="Training", unit="epoch", position=0, leave=True)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        logs = logs or {}
        self.progress_bar.set_postfix({
            'loss': f"{logs.get('loss', 0):.4f}",
            'val_loss': f"{logs.get('val_loss', 0):.4f}"
        })

    def on_train_end(self, logs=None):
        self.progress_bar.close()

async def collect_data(config, logger, coins=None):
    """Collect cryptocurrency data with progress bar."""
    logger.info("Starting data collection")

    data_config = config.get_data_config()
    selected_coins = coins if coins else data_config['coins']
    symbol_mapping = data_config.get('symbol_mapping', [])
    coin_map = data_config['coin_map']
    logger.info(f"Coins selected for data collection: {selected_coins}")

    collector = DataCollector(
        coins=selected_coins,
        days=data_config['days'],
        symbol_mapping=symbol_mapping,
        coin_map=coin_map,
    )

    async def collect_for_coin(coin):
        try:
            coin_data = await collector.collect_all_data([coin])
            if coin not in coin_data:
                logger.error(f"No data returned for {coin}")
                return coin, None
            sources = coin_data[coin]
            for source, df in sources.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info(f"Successfully collected data for {coin} from {source}")
                else:
                    logger.warning(f"Empty or unexpected data for {coin} from {source}")
            return coin, sources
        except Exception as e:
            logger.error(f"Failed to collect data for {coin}: {e}")
            return coin, None

    # Collect data concurrently
    tasks = [collect_for_coin(coin) for coin in selected_coins]
    results = await asyncio.gather(*tasks)

    data = {}
    for coin, coin_data in results:
        if coin_data:
            data[coin] = coin_data

    # Save collected data
    data_dir = Path(config.config['paths']['raw_data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    total_files = sum(len(sources) for sources in data.values())
    save_pbar = tqdm(total=total_files, desc="Saving data", unit="file")

    for coin, sources in data.items():
        for source, df in sources.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    filename = f"{coin}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
                    filepath = data_dir / filename
                    df.to_csv(filepath)
                    save_pbar.update(1)
                    logger.info(f"Saved {filename}")
                except Exception as e:
                    logger.error(f"Failed to save data for {coin} from {source}: {e}")
            else:
                logger.error(
                    f"Unexpected or empty data for {coin} from {source}. Data type: {type(df)}."
                )
    save_pbar.close()
    return data

def preprocess_data(data, config, logger):
    """
    Preprocess the collected data and save to the processed directory.
    """
    logger.info("Starting data preprocessing")

    model_config = config.get_model_config()
    data_config = config.get_data_config()

    processed_dir = Path(config.config['paths']['processed_data_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_data = {}

    for coin, sources in tqdm(data.items(), desc="Preprocessing", unit="coin"):
        if 'binance' in sources and not sources['binance'].empty:
            df = sources['binance']
        elif 'coingecko' in sources and not sources['coingecko'].empty:
            logger.warning(f"Using CoinGecko data for {coin} due to missing Binance data.")
            df = sources['coingecko']
        else:
            logger.error(f"No valid data found for {coin}. Skipping preprocessing.")
            continue

        pipeline = Pipeline(
            sequence_length=model_config['sequence_length'],
            prediction_length=model_config['prediction_length'],
            test_size=data_config.get('test_split', 0.2),
            validation_size=data_config.get('validation_split', 0.2)
        )

        try:
            coin_data = pipeline.run(df, save_dir=None)
        except Exception as e:
            logger.error(f"Error during preprocessing for {coin}: {e}")
            continue

        if config.config['preprocessing'].get('augment', False):
            coin_data['X_train'], coin_data['y_train'] = pipeline.augment_data(
                coin_data['X_train'],
                coin_data['y_train'],
                noise_level=config.config['preprocessing'].get('gaussian_noise', 0.01),
                shift_range=int(config.config['preprocessing'].get('sequence_overlap', 0.5) * pipeline.sequence_length)
            )
        else:
            logger.info("Data augmentation is disabled.")

        try:
            coin_dir = processed_dir / coin
            coin_dir.mkdir(parents=True, exist_ok=True)
            pipeline.save_processed_data(coin_data, str(coin_dir))
            processed_data[coin] = coin_data

            logger.info(f"Processed data for {coin}:")
            logger.info(f"X_train shape: {coin_data['X_train'].shape}")
            logger.info(f"y_train shape: {coin_data['y_train'].shape}")
            logger.info(f"X_val shape: {coin_data['X_val'].shape}")
            logger.info(f"y_val shape: {coin_data['y_val'].shape}")
            logger.info(f"X_test shape: {coin_data['X_test'].shape}")
            logger.info(f"y_test shape: {coin_data['y_test'].shape}")

            logger.info(f"Saved preprocessed data for {coin} to {coin_dir}")
        except Exception as e:
            logger.error(f"Failed to save processed data for {coin}: {e}")

    return processed_data

def create_visualizations(data, results, config, logger):
    """Create and save visualizations with progress bar."""
    logger.info("Creating visualizations")

    visualizer = CryptoVisualizer()
    viz_dir = Path(config.config['paths']['visualization_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)

    viz_pbar = tqdm(results.items(), desc="Creating visualizations", unit="coin")

    for coin, result in viz_pbar:
        viz_pbar.set_description(f"Creating visualizations for {coin}")

        # Training history
        fig_training = visualizer.plot_training_history(result['history'])
        visualizer.save_plot(fig_training, viz_dir / f"{coin}_training_history.html")

        # Performance metrics if available
        if 'evaluation' in result:
            fig_metrics = visualizer.plot_performance_metrics(result['evaluation'])
            visualizer.save_plot(fig_metrics, viz_dir / f"{coin}_performance_metrics.html")

        # Technical indicators
        fig_technical = visualizer.plot_technical_indicators(data[coin]['binance'])
        visualizer.save_plot(fig_technical, viz_dir / f"{coin}_technical_indicators.html")

        # Correlation matrix
        fig_correlation = visualizer.plot_correlation_matrix(data[coin]['binance'])
        visualizer.save_plot(fig_correlation, viz_dir / f"{coin}_correlation_matrix.html")

        logger.info(f"Saved visualizations for {coin}")

def save_results(results, config, logger):
    """Save training results and metrics with progress bar."""
    logger.info("Saving results")

    results_dir = Path(config.config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_pbar = tqdm(results.items(), desc="Saving results", unit="coin")

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, list)):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    for coin, result in save_pbar:
        save_pbar.set_description(f"Saving results for {coin}")
        result_path = results_dir / f"{coin}_results_{timestamp}.json"

        serializable_result = convert_to_serializable(result)
        with open(result_path, 'w') as f:
            json.dump(serializable_result, f, indent=4)

        logger.info(f"Saved results for {coin}")

async def predict_model(config, logger, coins=None, coin_name=None, latest=True, model_path=None):
    logger.info("Starting prediction process")

    models_dir = Path(config.config["paths"]["models_dir"])
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
    elif latest:
        model_files = sorted(models_dir.glob("*.keras"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not model_files:
            logger.error("No models found in the models directory")
            raise FileNotFoundError("No models available in the models directory.")
        model_path = model_files[0]
    else:
        raise ValueError("Either a model path or 'latest' flag must be provided.")

    logger.info(f"Using model: {model_path}")

    # Load the selected model
    model = CryptoPredictor.load(model_path)

    selected_coins = coins or ([coin_name] if coin_name else config.get_data_config()['coins'])
    logger.info(f"Selected coins for prediction: {selected_coins}")

    # Collect recent data for predictions
    collector = DataCollector(
        coins=selected_coins,
        days=30,
        symbol_mapping=config.get_data_config().get('symbol_mapping', []),
        coin_map=config.get_data_config()['coin_map']
    )
    raw_predict_dir = config.config['paths']['raw_predict_dir']

    logger.info("Fetching recent data for predictions.")
    await collector.collect_recent_data(selected_coins, 30, raw_predict_dir)

    # Preprocess recent data
    logger.info("Preprocessing recent data.")
    pipeline = Pipeline(
        sequence_length=config.get_model_config()['sequence_length'],
        prediction_length=config.get_model_config()['prediction_length']
    )

    recent_data_frames = []
    for file in Path(raw_predict_dir).glob("*.csv"):
        if coin_name and coin_name not in file.name:
            continue
        recent_data_frames.append(pd.read_csv(file))

    if not recent_data_frames:
        logger.error(f"No recent data files found for coin {coin_name or 'all coins'} in {raw_predict_dir}")
        raise FileNotFoundError(f"No recent data files found in {raw_predict_dir}")

    recent_data_combined = pd.concat(recent_data_frames, ignore_index=True)
    logger.info(f"Loaded recent data shape: {recent_data_combined.shape}")

    processed_recent_data = pipeline.transform(recent_data_combined)

    # Generate predictions
    logger.info("Generating predictions.")
    predictions = model.predict(processed_recent_data['X'])

    # Save predictions
    output_path = Path(config.config['paths']['predictions_output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving predictions to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(predictions.tolist(), f, indent=4)

    logger.info("Prediction process completed successfully.")

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (overrides config file console_level)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'collect-data', 'full-pipeline'],
                        help='Pipeline mode: train, predict, collect-data, or full-pipeline')
    parser.add_argument('--coins', type=str, nargs='*', default=None,
                        help='List of coins to process (overrides config file)')
    parser.add_argument("--coin-name", type=str, help="Coin name for prediction")
    parser.add_argument("--latest", action="store_true", help="Use the latest model")
    parser.add_argument("--model", type=str, help="Path to a specific model file")
    args = parser.parse_args()

    # Initialize a temporary logger
    logger = logging.getLogger('default')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    # Load configuration
    config = Config(args.config)

    # Setup logging
    try:
        logger_config = config.get_logging_config()
        log_level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        console_level = log_level_mapping[logger_config['console_level']]
        if args.log_level:
            console_level = log_level_mapping[args.log_level]

        logger = setup_logger(
            name=logger_config['name'],
            log_dir=logger_config['log_dir'],
            console_level=console_level,
            file_level=log_level_mapping[logger_config['file_level']],
            rotation=logger_config['rotation'],
            json_format=logger_config['json_format']
        )
    except ValueError as e:
        logger.error(f"Logging configuration error: {e}")
        sys.exit(1)

    logger.info(f"Pipeline starting in mode: {args.mode}")
    logger.info(f"Using configuration file: {args.config}")
    if args.coins:
        logger.info(f"User-selected coins: {args.coins}")
    else:
        logger.info("Using coins from configuration file.")

    # Validate configuration with the mode
    try:
        config.validate_config(mode=args.mode)
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)

    try:
        if args.mode == 'collect-data':
            # Collect data only
            data = await collect_data(config, logger, coins=args.coins)
            logger.info("Data collection completed successfully.")

        elif args.mode == 'train':
            # Full training pipeline
            logger.info("Starting training pipeline...")
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
            data = await collect_data(config, logger, coins=args.coins)
            processed_data = preprocess_data(data, config, logger)

            model_config = config.get_model_config()
            training_config = config.get_training_config()

            results = {}
            for coin, coin_data in processed_data.items():
                # Create model
                model = CryptoPredictor(
                    input_shape=(model_config['sequence_length'], coin_data['X_train'].shape[2]),
                    lstm_units=model_config['lstm_units'],
                    dropout_rate=model_config['dropout_rate'],
                    dense_units=model_config['dense_units'],
                    learning_rate=model_config['learning_rate'],
                    attention_units=model_config['attention_units']
                )
                model.build()
                model.compile()

                # Create trainer
                trainer = ModelTrainer(
                    model=model,
                    config=config,
                    batch_size=training_config['batch_size'],
                    epochs=training_config['epochs'],
                    early_stopping_patience=training_config['early_stopping_patience'],
                    reduce_lr_patience=training_config['reduce_lr_patience'],
                    min_delta=training_config['min_delta'],
                    model_dir=Path(config.config['paths']['models_dir'])
                )

                # Add progress callback
                trainer.add_callback(ProgressCallback(epochs=training_config['epochs']))

                # Train
                history = trainer.train(
                    X_train=coin_data['X_train'],
                    y_train=coin_data['y_train'],
                    X_val=coin_data['X_val'],
                    y_val=coin_data['y_val']
                )

                # Optional: Evaluate after training
                # evaluation = trainer.evaluate(coin_data['X_test'], coin_data['y_test'])

                # Predictions on test set
                predictions = trainer.predict(coin_data['X_test']).tolist()

                # Save model
                trainer.save_model(Path(config.config['paths']['models_dir']) / f"{coin}_model.keras", coin)

                # Store results
                results[coin] = {
                    'history': history.history,
                    # 'evaluation': evaluation, # Uncomment if evaluating
                    'predictions': predictions,
                    'actual_prices': coin_data['y_test'].tolist()
                }

            create_visualizations(data, results, config, logger)
            save_results(results, config, logger)

            logger.info("Training pipeline completed successfully.")

        elif args.mode == 'predict':
            # Run predictions
            logger.info("Starting prediction process...")
            await predict_model(
                config=config,
                logger=logger,
                coin_name=args.coin_name,
                latest=args.latest,
                model_path=args.model
            )
            logger.info("Prediction process completed successfully.")

        elif args.mode == 'full-pipeline':
            logger.info("Running full pipeline...")
            # Step 1: Collect data
            logger.info("Step 1: Collecting data")
            data = await collect_data(config, logger, coins=args.coins)
            # Step 2: Preprocess data
            logger.info("Step 2: Preprocessing data")
            processed_data = preprocess_data(data, config, logger)

            model_config = config.get_model_config()
            training_config = config.get_training_config()

            results = {}
            for coin, coin_data in processed_data.items():
                model = CryptoPredictor(
                    input_shape=(model_config['sequence_length'], coin_data['X_train'].shape[2]),
                    lstm_units=model_config['lstm_units'],
                    dropout_rate=model_config['dropout_rate'],
                    dense_units=model_config['dense_units'],
                    learning_rate=model_config['learning_rate'],
                    attention_units=model_config['attention_units']
                )
                model.build()
                model.compile()

                trainer = ModelTrainer(
                    model=model,
                    config=config,
                    batch_size=training_config['batch_size'],
                    epochs=training_config['epochs'],
                    early_stopping_patience=training_config['early_stopping_patience'],
                    reduce_lr_patience=training_config['reduce_lr_patience'],
                    min_delta=training_config['min_delta'],
                    model_dir=Path(config.config['paths']['models_dir'])
                )

                trainer.add_callback(ProgressCallback(epochs=training_config['epochs']))

                history = trainer.train(
                    X_train=coin_data['X_train'],
                    y_train=coin_data['y_train'],
                    X_val=coin_data['X_val'],
                    y_val=coin_data['y_val']
                )

                predictions = trainer.predict(coin_data['X_test']).tolist()
                # evaluation = trainer.evaluate(coin_data['X_test'], coin_data['y_test'])

                trainer.save_model(Path(config.config['paths']['models_dir']) / f"{coin}_model.keras", coin)

                results[coin] = {
                    'history': history.history,
                    # 'evaluation': evaluation,
                    'predictions': predictions,
                    'actual_prices': coin_data['y_test'].tolist()
                }

            # Step 4: Save results
            logger.info("Step 4: Saving results")
            save_results(results, config, logger)

            # Step 5: Make predictions
            logger.info("Step 5: Making predictions")
            for coin in args.coins or config.get_data_config()["coins"]:
                await predict_model(
                    config=config,
                    logger=logger,
                    coin_name=coin,
                    latest=True
                )

            logger.info("Full pipeline completed successfully.")

        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
