#!/bin/bash
# Run evaluation on all videos in a directory at specified FPS settings
#
# Usage:
#   ./scripts/run_eval.sh <data_dir> [fps1] [fps2] ...
#
# Examples:
#   ./scripts/run_eval.sh /path/to/labeled_data 10 5
#   ./scripts/run_eval.sh /path/to/labeled_data 0    # native FPS

set -e

DATA_DIR="${1:-.}"
shift || true  # shift past data dir, allow empty

# Default FPS values if none provided
if [ $# -eq 0 ]; then
    FPS_VALUES=(0)  # native FPS
else
    FPS_VALUES=("$@")
fi

# Build the evaluate binary first
echo "Building evaluate binary..."
cargo build --bin evaluate --release

# Find all human-labeled JSON files (look for files with "detections" in name or check for required fields)
find_labeled_files() {
    local dir="$1"
    # Look for pairs of detection + timestamp files
    # Pattern 1: same directory with *_detections.json and *_timestamps.json
    # Pattern 2: detections.json and timestamps.json in subdirectories

    # Try to find detection files with corresponding timestamp files
    find "$dir" -name "*.json" -type f | while read -r det_file; do
        # Check if this looks like a detection file (has "detections" array)
        if grep -q '"detections"' "$det_file" 2>/dev/null && grep -q '"video_path"' "$det_file" 2>/dev/null; then
            # This is a human-labeled detection file
            # Look for corresponding timestamps file
            base_dir=$(dirname "$det_file")
            base_name=$(basename "$det_file" .json)

            # Try common timestamp file patterns
            for ts_pattern in \
                "${base_dir}/timestamps.json" \
                "${base_dir}/${base_name}_timestamps.json" \
                "${base_dir}/timestamps_${base_name}.json" \
                "${base_dir}/../timestamps.json" \
                "${det_file%.json}_timestamps.json"; do
                if [ -f "$ts_pattern" ]; then
                    echo "$det_file|$ts_pattern"
                    break
                fi
            done
        fi
    done
}

# Also handle trio-warehouse format (directories with video_*, timestamps_*, detections_*)
find_trio_dirs() {
    local dir="$1"
    find "$dir" -type d | while read -r subdir; do
        if ls "$subdir"/detections_*.json "$subdir"/timestamps_*.json &>/dev/null; then
            det_file=$(ls "$subdir"/detections_*.json 2>/dev/null | head -1)
            ts_file=$(ls "$subdir"/timestamps_*.json 2>/dev/null | head -1)
            if [ -n "$det_file" ] && [ -n "$ts_file" ]; then
                echo "$det_file|$ts_file"
            fi
        fi
    done
}

echo ""
echo "Scanning for data files in: $DATA_DIR"
echo "FPS settings to evaluate: ${FPS_VALUES[*]}"
echo ""

# Collect all video pairs
VIDEO_PAIRS=()
while IFS= read -r line; do
    [ -n "$line" ] && VIDEO_PAIRS+=("$line")
done < <(find_labeled_files "$DATA_DIR")

while IFS= read -r line; do
    [ -n "$line" ] && VIDEO_PAIRS+=("$line")
done < <(find_trio_dirs "$DATA_DIR")

# Remove duplicates
VIDEO_PAIRS=($(printf '%s\n' "${VIDEO_PAIRS[@]}" | sort -u))

if [ ${#VIDEO_PAIRS[@]} -eq 0 ]; then
    echo "No data files found in $DATA_DIR"
    echo ""
    echo "Expected format:"
    echo "  Human-labeled: <video>_detections.json + timestamps.json"
    echo "  Trio-warehouse: directory with detections_*.json + timestamps_*.json"
    exit 1
fi

echo "Found ${#VIDEO_PAIRS[@]} video(s) to evaluate"
echo ""

# Run evaluations
for pair in "${VIDEO_PAIRS[@]}"; do
    IFS='|' read -r det_file ts_file <<< "$pair"

    video_name=$(basename "$det_file" .json)
    echo "========================================"
    echo "Video: $video_name"
    echo "  Detections: $det_file"
    echo "  Timestamps: $ts_file"
    echo ""

    for fps in "${FPS_VALUES[@]}"; do
        if [ "$fps" = "0" ]; then
            echo "--- Native FPS ---"
        else
            echo "--- ${fps} FPS ---"
        fi

        ./target/release/evaluate "$det_file" "$ts_file" --tracker-fps "$fps"
        echo ""
    done
done

echo "========================================"
echo "All evaluations complete!"
