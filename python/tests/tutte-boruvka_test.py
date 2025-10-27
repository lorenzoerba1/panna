import numpy as np
import pandas as pd
import time
import fast_hdbscan

if __name__ == "__main__":
    # DImensionalities to test
    dim_size = [10, 100, 200, 400, 800, 1200]
    # Number of samples to test
    size = [10**4, 10**5, 10**6]
    result = pd.DataFrame(columns=["Algorithm", "n", "D", "Time (s)"])
    
    for dim in dim_size:
        for n in size:
            print(f"Data size: {n}, Dimension: {dim}")
        # Generate random data
            data = np.random.rand(n, dim).astype(np.float32)

            # Run Fast HDBSCAN
            
            start_time = time.perf_counter()
            mst = fast_hdbscan.HDBSCAN(min_samples=1).fit(data)
            end_time = time.perf_counter()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            result = pd.concat([result, pd.DataFrame({
                "Algorithm": ["Tutte-Boruvka"],
                "n": [n],
                "D": [dim],
                "Time (s)": [end_time - start_time]
            })], ignore_index=True)
    
    result.to_csv("tutte_boruvka_performance.csv", index=False)