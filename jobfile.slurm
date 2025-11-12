#!/bin/bash
#SBATCH -J panna_sequential
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH --partition lovelace
#SBATCH --mem 0
#SBATCH --exclusive
#SBATCH --time 5-00:00:00 # This is the global time limit for the job, increased to 5 days hoping that it can accomodate all timeouts of the baselines

TEST_TIME_LIMIT=480m
DOWNLOAD_DIR="datasets"
DATASETS_URLS= (
    "https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/landmark-nomic-768-normalized.hdf5?download=true",
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5?download=true",
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5?download=true",
    "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
    "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
    "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz",
    "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
    "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip"
)

# Create datasets and resutls directories if they don't exist
mkdir -p $DOWNLOAD_DIR
mkdir -p results
echo "Datasets and results directories are set up."
# Download datasets if they are not already present
for url in "${DATASETS_URLS[@]}"
do
    filename=$(basename "$url")
    if [ ! -f "$DOWNLOAD_DIR/$filename" ]; then
        echo "Downloading $filename..."
        wget -P "$DOWNLOAD_DIR" "$url"
    fi

done
echo "All datasets are downloaded."

# Unzip the gas and chem datasets
unzip -o "$DOWNLOAD_DIR/gas+sensors+for+home+activity+monitoring.zip" -d "$DOWNLOAD_DIR"
unzip -o "$DOWNLOAD_DIR/gas+sensor+array+under+dynamic+gas+mixtures.zip" -d "$DOWNLOAD_DIR"
unzip -o "$DOWNLOAD_DIR/HT_Sensor_dataset.zip" -d "$DOWNLOAD_DIR"


export APPTAINER_BIND="results:/results,datasets:/datasets,tests:/tests"

# Loop through all .py files in the tests folder and run them sequentially
for test_file in tests/*.py
do
    echo "--- Running test: $test_file ---"
    timeout "$TEST_TIME_LIMIT" srun panna.sif "$test_file"
    
    if [ $? -eq 124 ]; then
        echo "ERROR: Test $test_file timed out after $TEST_TIME_LIMIT."
    elif [ $? -ne 0 ]; then
        echo "ERROR: Test $test_file failed."
    fi
    echo "--- Finished test: $test_file ---"
done

echo "All tests completed."