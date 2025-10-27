import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
import panna

if __name__ == "__main__":
    dimensions = [50, 100, 200, 400, 800, 1600] #, 128, 1024]
    size = [10**5]#, 10**6]
    
    with open("scalability_results_toc.csv", "a+") as f_out:
        for dim in dimensions:
            for sz in size:
                data = np.random.rand(sz, dim).astype(np.float32)
                
                start = perf_counter()
                panna.emst_theory_of_computing(data)
                elapsed_time = perf_counter() - start
                
                print(f"Dimension: {dim}, Size: {sz}, Time: {elapsed_time:.2f} seconds")
                f_out.write(f"ToC-Kr,{sz},{dim},{elapsed_time}\n")