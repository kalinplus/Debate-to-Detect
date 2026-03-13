#!/bin/bash
# Run batch detection - edit parameters below, then run: ./run_batch.sh

# Data source: main, m4, detectrl_multidomain, detectrl_multillm, raid, text_attack, realdet, base, test
DATA_SOURCE="test"

# Model to use: gpt-4o-mini, gpt-4o, gpt-4, etc.
MODEL="gpt-5.1"

# Maximum samples to process (-1 for all)
MAX_SAMPLES="-1"

# Temperature (0.0-2.0, higher = more creative)
TEMPERATURE="1.0"

# Sleep time between API calls (seconds)
SLEEP="0.5"

# Output directory
OUTPUT_DIR="Results/quick_test1"

# Optional: Dataset-specific parameters
# DATASET="xsum"              # For main data source
# SOURCE_MODEL="gpt4o"        # For main/text_attack data source
# ATTACK_TYPE="delete"        # For text_attack data source
# BASE_DATASET="xsum"         # For base data source
# BASE_SOURCE_MODEL="gpt-j-6B" # For base data source

# Run batch detection
python batch_detect.py \
    --data-source "$DATA_SOURCE" \
    --model "$MODEL" \
    --max-samples "$MAX_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --sleep "$SLEEP" \
    --output-dir "$OUTPUT_DIR"
    # Uncomment below if using specific data sources
    # --dataset "$DATASET" \
    # --source-model "$SOURCE_MODEL" \
    # --attack-type "$ATTACK_TYPE" \
    # --base-dataset "$BASE_DATASET" \
    # --base-source-model "$BASE_SOURCE_MODEL"

