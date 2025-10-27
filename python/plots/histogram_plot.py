import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path

if __name__ == "__main__":
    mpl.use("WebAgg")
    # Read the CSV file
    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "histogram.csv")
    df = pd.read_csv(filepath)
    sns.set_theme(palette="muted", style="white")
    
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
    sns.barplot(data=df, x="weight", y="count", ax=axs)
    
    plt.show()