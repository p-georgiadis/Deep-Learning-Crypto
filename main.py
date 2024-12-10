# main.py

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_collection.data_collector import DataCollector
from src.preprocessing.pipeline import Pipeline
from src.training.model import CryptoPredictor
from src.training.trainer import ModelTrainer
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.visualizer import CryptoVisualizer
from src.utils.custom_losses import di_mse_loss, directional_accuracy
from src.utils.callbacks import DirectionWeightCallback

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


async def collect_data(config: Config, logger: logging.Logger, coins: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    logger.info("Starting data collection.")
    data_config = config.get_data_config()
    selected_coins = coins or data_config.get('coins', [])
    logger.info(f"Selected coins: {selected_coins}")

    raw_data_dir = Path(config.get_path('raw_data_dir'))
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    existing_data = {}
    coins_to_fetch = []

    for coin in selected_coins:
        binance_file = raw_data_dir / f"{coin}_binance_{today_str}.csv"
        if binance_file.exists():
            # If we have binance data for today, skip fetching
            df = pd.read_csv(binance_file, index_col=0)
            existing_data[coin] = {'binance': df}
            logger.info(f"Data for {coin} already exists. Skipping collection.")
        else:
            coins_to_fetch.append(coin)

    if not coins_to_fetch:
        logger.info("All selected coins already have today's data. Exiting data collection successfully.")
        return existing_data

    collector = DataCollector(
        coins=coins_to_fetch,
        days=data_config['days'],
        symbol_mapping=[],  # Adjust symbol_mapping as needed in config
        coin_map=data_config.get('coin_map', {})
    )

    data = await collector.collect_all_data(coins_to_fetch)

    for coin, sources in data.items():
        for source, df in sources.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{coin}_{source}_{today_str}.csv"
                df.to_csv(raw_data_dir / filename)
                logger.info(f"Saved {filename} for {coin}.")
            else:
                logger.warning(f"No data for {coin} from {source}.")

    all_data = {**existing_data, **data}
    logger.info("Data collection completed.")
    return all_data


def preprocess_and_train(config: Config, logger: logging.Logger, data: Dict[str, Dict[str, pd.DataFrame]]):
    logger.info("Starting preprocessing and training.")
    processed_data = {}
    pipelines = {}

    processed_dir = Path(config.get_path('processed_data_dir'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    for coin, sources in data.items():
        # We now only have binance data
        binance_df = sources.get('binance')

        if binance_df is not None and not binance_df.empty:
            df = binance_df
        else:
            logger.warning(f"No valid data for {coin}, skipping.")
            continue

        pipeline = Pipeline(config=config)
        # Pass save_dir to ensure scalers are saved
        coin_data = pipeline.run(df, save_dir=str(processed_dir / coin))
        processed_data[coin] = coin_data
        pipelines[coin] = pipeline
        logger.info(f"Preprocessed data for {coin}")

    results = {}
    model_config = config.get_model_config()
    training_config = config.get_training_config()

    for coin, coin_data in processed_data.items():
        logger.info(f"Training model for {coin}...")

        input_shape = (model_config["sequence_length"], coin_data["X_train"].shape[2])
        model = CryptoPredictor(
            input_shape=input_shape,
            lstm_units=model_config['lstm_units'],
            dropout_rate=model_config['dropout_rate'],
            dense_units=model_config['dense_units'],
            learning_rate=model_config['learning_rate'],
            clip_norm=model_config.get('clip_norm', None)
        )

        model.build()
        model.compile(loss=di_mse_loss)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        direction_weight_cb = DirectionWeightCallback(total_epochs=training_config['epochs'])

        trainer = ModelTrainer(
            model=model.model,
            model_dir=config.get_path('models_dir'),
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            early_stopping_patience=training_config['early_stopping']['patience'],
            min_delta=training_config['early_stopping']['min_delta']
        )

        history = trainer.train(
            X_train=coin_data["X_train"],
            y_train=coin_data["y_train"],
            X_val=coin_data["X_val"],
            y_val=coin_data["y_val"],
            additional_callbacks=[reduce_lr, direction_weight_cb]
        )

        eval_results = model.evaluate(coin_data["X_test"], coin_data["y_test"])
        preds_scaled = model.predict(coin_data["X_test"])

        preds_original = pipelines[coin].inverse_transform_predictions(preds_scaled)
        y_test_original = pipelines[coin].inverse_transform_actuals(coin_data["y_test"])

        coin_model_dir = Path(config.get_path('models_dir')) / coin
        coin_model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = coin_model_dir / "model.keras"
        model.save(model_save_path)

        results[coin] = {
            "history": history.history,
            "evaluation": eval_results,
            "predictions": preds_original.tolist(),
            "actual_prices": y_test_original.tolist()
        }

    return results


def save_results(results: Dict[str, Dict], config: Config, logger: logging.Logger):
    logger.info("Saving results.")
    results_dir = Path(config.get_path('results_dir'))
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for coin, result in results.items():
        result_path = results_dir / f"{coin}_results_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved results for {coin} at {result_path}")

async def run_prediction(config: Config, logger: logging.Logger, coins: Optional[List[str]] = None):
    logger.info("Starting prediction mode...")
    data_config = config.get_data_config()
    selected_coins = coins or data_config.get('coins', [])
    logger.info(f"Selected coins for prediction: {selected_coins}")

    raw_predict_dir = Path(config.get_path('raw_predict_dir'))
    raw_predict_dir.mkdir(parents=True, exist_ok=True)

    days_back = 100
    collector = DataCollector(
        coins=selected_coins,
        days=days_back,
        symbol_mapping=[],
        coin_map=data_config.get('coin_map', {})
    )

    today_str = datetime.now().strftime('%Y%m%d')
    all_predictions = {}

    # Create the results/predictions directory
    prediction_output_dir = Path(config.get_path('results_dir')) / "predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    for coin in selected_coins:
        recent_file = raw_predict_dir / f"{coin}_binance_recent_{today_str}.csv"
        if not recent_file.exists():
            logger.info(f"Recent data for {coin} not found. Collecting now...")
            await collector.collect_recent_data([coin], days_back, raw_predict_dir)

        if not recent_file.exists():
            logger.error(f"Failed to collect recent data for {coin}, skipping prediction.")
            continue

        df = pd.read_csv(recent_file, index_col=0)
        if df.empty:
            logger.warning(f"Recent data for {coin} is empty, skipping.")
            continue

        pipeline = Pipeline(config=config)

        processed_dir = Path(config.get_path('processed_data_dir')) / coin
        scaler_dir = processed_dir / "scalers"
        if not scaler_dir.exists():
            logger.error(f"No scaler directory found for {coin}. Cannot predict.")
            continue

        pipeline.load_scaler(
            scaler_path=scaler_dir / "feature_scaler.joblib",
            target_scaler_path=scaler_dir / "target_scaler.joblib"
        )

        numeric_features_path = processed_dir / "numeric_features.json"
        if numeric_features_path.exists():
            with open(numeric_features_path, 'r') as f:
                numeric_features = json.load(f)
            pipeline.numeric_features = numeric_features
        else:
            logger.error(f"No numeric_features.json found for {coin}, cannot proceed.")
            continue

        # Run pipeline in prediction mode to get the last sequence
        prediction_data = pipeline.run(df, prediction_mode=True)
        X = prediction_data['X']

        model_path = Path(config.get_path('models_dir')) / coin / "model.keras"
        logger.info(f"Loading model for coin='{coin}' from '{model_path}'...")
        if not model_path.exists():
            logger.error(f"No trained model found for {coin} at {model_path}. Skipping prediction.")
            continue

        model = CryptoPredictor.load(model_path)

        # Number of future days to predict
        future_days = 5
        predictions = []
        current_X = X.copy()

        # 'close' should be the last column after pipeline processing
        close_idx = len(pipeline.numeric_features)  # 'close' is appended as the last column

        for i in range(future_days):
            preds_scaled = model.predict(current_X)
            preds_original = pipeline.inverse_transform_predictions(preds_scaled)  # real-world price
            predicted_price = float(preds_original[0])
            predictions.append(predicted_price)

            # Re-transform predicted real-world price back into target-scaled domain
            rescaled_close = pipeline.target_scaler.transform(np.array([[predicted_price]]))[0, 0]

            # Update sequence with the newly predicted day
            new_row = current_X[0, -1, :].copy()
            new_row[close_idx] = rescaled_close

            # Shift and update sequence for next prediction
            current_X = np.roll(current_X, -1, axis=1)
            current_X[0, -1, :] = new_row

        # Create a user-friendly JSON structure
        prediction_output = {
            "coin": coin,
            "prediction_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_horizon_days": future_days,
            "predictions": [
                {"day": i+1, "expected_price": predictions[i]} for i in range(future_days)
            ],
            "explanation": (
                f"These predictions represent the model's best estimate of {coin} closing prices "
                f"for the next {future_days} days, based on historical patterns and recent data."
            )
        }

        # Save predictions in results/predictions/{coin}_future_predictions.json
        coin_prediction_path = prediction_output_dir / f"{coin}_future_predictions.json"
        with open(coin_prediction_path, 'w') as f:
            json.dump(prediction_output, f, indent=4)
        logger.info(f"{future_days}-day future predictions for {coin} saved to {coin_prediction_path}")

        all_predictions[coin] = prediction_output

    logger.info("Prediction mode completed.")

async def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "predict", "collect-data", "full-pipeline"],
                        help="Pipeline mode: train, predict, collect-data, or full-pipeline")
    parser.add_argument("--coins", type=str, nargs="*", default=None,
                        help="List of coins to process (overrides config file)")
    args = parser.parse_args()

    temp_logger = logging.getLogger("default")
    temp_logger.addHandler(logging.StreamHandler(sys.stdout))
    temp_logger.setLevel(logging.INFO)

    try:
        config = Config(args.config)
    except Exception as e:
        temp_logger.error(f"Error loading config: {e}")
        sys.exit(1)

    logger_config = config.get_logging_config()
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    console_level = log_level_mapping.get(logger_config.get("console_level", "INFO"), logging.INFO)
    file_level = log_level_mapping.get(logger_config.get("file_level", "DEBUG"), logging.DEBUG)
    logger = setup_logger(
        name=logger_config["name"],
        log_dir=logger_config["log_dir"],
        console_level=console_level,
        file_level=file_level,
        rotation=logger_config["rotation"]
    )

    logger.info(f"Running in mode: {args.mode}")

    if args.mode == "collect-data":
        data = await collect_data(config, logger, coins=args.coins)
        logger.info("Data collection completed.")
    elif args.mode == "train":
        data = await collect_data(config, logger, coins=args.coins)
        results = preprocess_and_train(config, logger, data)
        save_results(results, config, logger)
        logger.info("Training completed.")
    elif args.mode == "predict":
        await run_prediction(config, logger, coins=args.coins)
    elif args.mode == "full-pipeline":
        data = await collect_data(config, logger, coins=args.coins)
        results = preprocess_and_train(config, logger, data)
        save_results(results, config, logger)
        await run_prediction(config, logger, coins=args.coins)
        logger.info("Full pipeline completed.")
    else:
        logger.warning("Unknown mode selected.")

if __name__ == "__main__":
    asyncio.run(main())