#!/bin/bash
#SBATCH -J panna_sequential
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH --partition lovelace
#SBATCH --mem 0
#SBATCH --exclusive
#SBATCH --time 5-00:00:00 # This is the global time limit for the job, increased to 5 days hoping that it can accomodate all timeouts of the baselines

TEST_TIME_LIMIT=480m

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