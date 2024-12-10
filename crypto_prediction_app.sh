#!/bin/bash

# Check required dependencies
for tool in yq jq python; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: '$tool' is not installed. Please install it before running this script."
        exit 1
    fi
done

choose_coins() {
    echo "Available coins from config.yaml:"
    coin_list=$(yq '.data.coins[]' "$CONFIG_FILE" 2>/dev/null)

    if [ -z "$coin_list" ]; then
        echo "Error: No coins found in configuration (.data.coins). Please add coins to config.yaml."
        exit 1
    fi

    coin_array=()
    index=1
    while IFS= read -r coin; do
        echo "$index) $coin"
        coin_array+=("$coin")
        index=$((index + 1))
    done <<< "$coin_list"

    echo "$index) All coins"
    read -p "Enter the numbers corresponding to the coins to process (comma-separated, or $index for all): " coins_input

    if [[ "$coins_input" =~ ^[0-9]+$ ]] && [ "$coins_input" -eq "$index" ]; then
        COINS="${coin_array[*]}"
    else
        COINS=""
        IFS=',' read -ra selected_indices <<< "$coins_input"
        for idx in "${selected_indices[@]}"; do
            if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx > 0 && idx <= ${#coin_array[@]} )); then
                COINS="$COINS ${coin_array[$((idx - 1))]}"
            else
                echo "Invalid selection: $idx"
                exit 1
            fi
        done
        COINS=$(echo "$COINS" | xargs)
    fi
    echo "Selected coins: $COINS"
}

choose_config() {
    echo "Would you like to use the default config.yaml (configs/config.yaml) or specify a different one?"
    echo "1) Use default config.yaml"
    echo "2) Specify a different config.yaml"
    read -p "Enter your choice (1 or 2): " config_choice

    if [ "$config_choice" -eq 2 ]; then
        read -p "Enter the path to the configuration file: " custom_config
        if [ ! -f "$custom_config" ]; then
            echo "Error: Configuration file not found at $custom_config. Exiting."
            exit 1
        fi
        CONFIG_FILE="$custom_config"
    else
        CONFIG_FILE="configs/config.yaml"
    fi

    # Quick validation: ensure .paths key exists
    if ! yq '.paths' "$CONFIG_FILE" &>/dev/null; then
        echo "Error: The chosen config file does not have a '.paths' key or is invalid."
        exit 1
    fi

    echo "Using configuration file: $CONFIG_FILE"
}

choose_mode() {
    echo "Select a mode to run the application:"
    echo "1) Collect Data"
    echo "2) Train Model (includes data preprocessing)"
    echo "3) Make Predictions"
    echo "4) Run Full Pipeline"
    read -p "Enter your choice (1-4): " mode_choice

    case $mode_choice in
        1) MODE="collect-data";;
        2) MODE="train";;
        3) MODE="predict";;
        4) MODE="full-pipeline";;
        *) echo "Invalid choice. Exiting."; exit 1;;
    esac
}

run_mode() {
    echo "-------------------------------------"
    echo "Mode selected: $MODE"
    echo "Configuration file: $CONFIG_FILE"
    echo "Coins: ${COINS:-Default from config.yaml}"
    echo "-------------------------------------"

    read -p "Do you want to proceed? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Operation cancelled by the user. Exiting."
        exit 0
    fi

    echo "Running in $MODE mode..."
    if [[ "$MODE" == "predict" ]]; then
        if [ -z "$COINS" ]; then
            python main.py --config "$CONFIG_FILE" --mode "$MODE"
        else
            python main.py --config "$CONFIG_FILE" --mode "$MODE" --coins $COINS
        fi
    else
        if [ -z "$COINS" ]; then
            python main.py --config "$CONFIG_FILE" --mode "$MODE"
        else
            python main.py --config "$CONFIG_FILE" --mode "$MODE" --coins $COINS
        fi
    fi

    if [ $? -ne 0 ]; then
        echo "Error: The operation failed. Please check the logs for details."
        exit 1
    fi

    echo
    echo "Operation completed successfully. You can now choose another action or exit."
}

echo "Welcome to the Cryptocurrency Prediction App!"

while true; do
    echo
    echo "1) Choose a mode (e.g., Collect Data, Train Model) and execute"
    echo "2) View the current configuration (config.yaml)"
    echo "3) Exit the application"
    read -p "Please enter your choice (1-3): " choice

    case $choice in
        1)
            choose_config
            choose_coins
            choose_mode
            run_mode
            ;;
        2)
            echo "Displaying the current configuration:"
            cat configs/config.yaml
            ;;
        3)
            echo "Exiting the application. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done
