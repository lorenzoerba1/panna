#!/bin/bash
set -euo pipefail

# --- Configuration ---
DOWNLOAD_DIR="datasets2"
DATASETS_URLS=(
    "https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/landmark-nomic-768-normalized.hdf5?download=true"
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5?download=true"
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5?download=true"
    "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
    "http://ann-benchmarks.com/glove-100-angular.hdf5"
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    "http://ann-benchmarks.com/nytimes-256-angular.hdf5"
    "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz"
    "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip"
    "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip"
)

# Create download directory
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "Setting up $DOWNLOAD_DIR directory..."
    mkdir -p "$DOWNLOAD_DIR"
fi


# Download the datasets
echo "Starting dataset downloads..."
for url in "${DATASETS_URLS[@]}"
do
filename="${url##*/}"
    
    # Handle the specific Hugging Face URLs to remove query parameters like ?download=true
    if [[ "$filename" == *'?'* ]]; then
        # Remove the query part (everything from the '?' onwards)
        filename="${filename%%\?*}"
    fi

    if [ -z "$filename" ]; then
        echo "ERROR: Could not determine filename for URL: $url. Skipping."
        continue
    fi

    TARGET_PATH="$DOWNLOAD_DIR/$filename"
    
    if [ -f "$TARGET_PATH" ]; then
        echo "Skipping $filename: already exists."
    else
        echo "Downloading $filename to $DOWNLOAD_DIR/..."
        # Using wget to download. The -O flag ensures the name is correct.
        wget -q --show-progress -O "$TARGET_PATH" "$url"
        if [ $? -ne 0 ]; then
            echo "ERROR: Download of $filename failed! Please check the URL or your connection."
            exit 1
        fi
    fi
done
echo "All datasets checked and downloaded."

# --- Unzip Files ---
echo "Starting file extraction..."
# Files to unzip 
ZIP_FILES=(
    "gas+sensors+for+home+activity+monitoring.zip"
    "gas+sensor+array+under+dynamic+gas+mixtures.zip"
    "PAMAP2_Dataset.zip"
    "HT_Sensor_dataset.zip"  # Unzipped from gas+sensors+home
    # PAMAP2 is not unzipped, since the dataset needs to be created by appending multiple files
    # so I directly deal with this in the python files  
)

for zipfile in "${ZIP_FILES[@]}"; do
    ZIP_PATH="$DOWNLOAD_DIR/$zipfile"
    if [ -f "$ZIP_PATH" ]; then
        echo "Unzipping $zipfile..."
        unzip -o "$ZIP_PATH" -d "$DOWNLOAD_DIR"
    else
        echo "Warning: $zipfile not found. Skipping extraction."
    fi
done

echo "Extraction complete."
echo "--- Dataset setup finished successfully! ---"