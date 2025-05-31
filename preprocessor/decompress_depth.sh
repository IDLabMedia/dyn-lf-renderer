#!/bin/bash

# Exit on any error
set -e

# Check for input directory
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

INPUT_DIR="$1"

# Check if directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Process each *_depth.mp4 file in the directory
shopt -s nullglob
for FILE in "$INPUT_DIR"/*_depth.mp4; do
    BASENAME=$(basename "$FILE" .mp4)
    OUTPUT_FILE="$INPUT_DIR/${BASENAME}.yuv"

    echo "Decoding '$FILE' to '$OUTPUT_FILE'..."

    if ! ffmpeg -v error -i "$FILE" -f rawvideo -pix_fmt yuv420p10le "$OUTPUT_FILE" > /dev/null 2>&1; then
        echo "  Error decoding '$FILE'"
        exit 1
    fi

    echo "  Done."
done
shopt -u nullglob
