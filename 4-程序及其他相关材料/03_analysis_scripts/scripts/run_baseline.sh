#!/bin/bash

# EVVES-DDDQN Baseline Comparison Script
# Compares DDDQN performance against 4 baseline strategies

set -e  # Exit on error

echo "EVVES-DDDQN Baseline Comparison"
echo "==============================="
echo "Baseline strategies:"
echo "  1. Random policy"
echo "  2. Greedy economic policy"
echo "  3. SOC balancing policy"
echo "  4. Peak shaving policy"
echo ""

# Configuration
SEEDS=(0 1 2 3)
RESULTS_DIR="results"
MODELS_DIR="models"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if trained models exist
echo "Checking for trained models..."
missing_models=0
for seed in "${SEEDS[@]}"; do
    model_file="$MODELS_DIR/halfyear_seed${seed}.zip"
    if [ ! -f "$model_file" ]; then
        echo "Warning: Model not found: $model_file"
        missing_models=$((missing_models + 1))
    else
        echo "Found: $model_file"
    fi
done

if [ $missing_models -gt 0 ]; then
    echo ""
    echo "Warning: $missing_models model(s) missing."
    echo "Please run training first: scripts/run_train_halfyear.sh"
    echo "Continuing with available models..."
fi

echo ""
echo "Starting baseline comparison..."

# Run comparison for each seed
for seed in "${SEEDS[@]}"; do
    model_file="$MODELS_DIR/halfyear_seed${seed}.zip"
    
    if [ -f "$model_file" ]; then
        echo "Running comparison for seed $seed..."
        
        python code/baseline_comparison.py \
            --model_path "$model_file" \
            --seed $seed \
            --output_dir "$RESULTS_DIR" \
            --episodes 10 \
            --verbose
        
        if [ $? -eq 0 ]; then
            echo "Comparison completed for seed $seed"
        else
            echo "Comparison failed for seed $seed"
        fi
    else
        echo "Skipping seed $seed (model not found)"
    fi
    
    echo ""
done

# Aggregate results
echo "Aggregating results..."
python code/statistics.py \
    --results_dir "$RESULTS_DIR" \
    --output_file "$RESULTS_DIR/aggregated_comparison.csv" \
    --generate_plots

echo ""
echo "Baseline comparison completed!"
echo "Results saved to: $RESULTS_DIR/"

# Display summary
if [ -f "$RESULTS_DIR/aggregated_comparison.csv" ]; then
    echo ""
    echo "Performance Summary:"
    echo "==================="
    head -10 "$RESULTS_DIR/aggregated_comparison.csv"
fi

echo ""
echo "Generated files:"
find "$RESULTS_DIR" -name "*.csv" -o -name "*.png" -o -name "*.json" | sort 