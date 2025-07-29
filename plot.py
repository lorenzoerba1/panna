import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    mpl.use("WebAgg")
    sns.set_theme(style="white")
    # 200 random points with a planted sine motif at position 23 and 72
    x = 0.1*np.random.rand(300,3)
    x[23:43,0] += 0.125* np.sin(np.linspace(0, 4 * np.pi, 20)) 
    x[72:92,0] += 0.125 * np.sin(np.linspace(0, 4 * np.pi, 20))
    x[23:43,1] +=0.125* np.cos(np.linspace(0, 3.6 * np.pi, 20)) 
    x[72:92,1] += 0.125* np.cos(np.linspace(0, 3.6 * np.pi, 20))
    # A triangular motif at position 100 and 200, windows of 20 points, dimensions 1 and 2
    x[100:120,1] += 0.125 * np.linspace(0, 1, 20)
    x[200:220,1] += 0.125 * np.linspace(0, 1, 20)
    x[100:120,2] += 0.125 * np.linspace(0, 1, 20)
    x[200:220,2] += 0.125 * np.linspace(0, 1, 20)
    # A step motif at position 150 and 250, windows of 20 points, dimensions 0 and 2
    x[150:170,0] = 0.125 * np.ones(20) + 0.01 * np.random.rand(20)
    x[250:270,0] = 0.125 * np.ones(20) + 0.01 * np.random.rand(20)
    x[150:170,2] = 0.125 * np.ones(20) + 0.01 * np.random.rand(20)
    x[250:270,2] = 0.125 * np.ones(20) + 0.01 * np.random.rand(20)

    # Min max scaling
    x[:,0] = (x[:,0] - np.min(x[:,0])) / (np.max(x[:,0]) - np.min(x[:,0]))
    x[:,1] = (x[:,1] - np.min(x[:,1])) / (np.max(x[:,1]) - np.min(x[:,1]))
    x[:,2] = (x[:,2] - np.min(x[:,2])) / (np.max(x[:,2]) - np.min(x[:,2]))
    # Shift the data to
    
    x[:,1] -= 1.2
    x[:,0] += 0.8
    fig, ax = plt.subplots(figsize=(12, 4))

    sns.lineplot(x=np.arange(300), y=x[:,0], color='gray', alpha=0.6)
    sns.lineplot(x=np.arange(23, 43), y=x[23:43,0], color='forestgreen', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(72, 92), y=x[72:92,0], color='forestgreen', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(300), y=x[:,1], color='gray', alpha=0.6, linewidth=2)
    sns.lineplot(x=np.arange(300), y=x[:,2] , color='gray', alpha=0.6, linewidth=2)
    sns.lineplot(x=np.arange(23, 43), y=x[23:43,1], color='forestgreen', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(72, 92), y=x[72:92,1], color='forestgreen', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(100, 120), y=x[100:120,1], color='darkorange', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(200, 220), y=x[200:220,1], color='darkorange', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(100, 120), y=x[100:120,2], color='darkorange', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(200, 220), y=x[200:220,2], color='darkorange', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(150, 170), y=x[150:170,0], color='darkblue', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(250, 270), y=x[250:270,0], color='darkblue', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(150, 170), y=x[150:170,2], color='darkblue', linewidth=3, alpha=0.8)
    sns.lineplot(x=np.arange(250, 270), y=x[250:270,2], color='darkblue', linewidth=3, alpha=0.8)
    # remove grid and spines, just the plot
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axes().set_axis_off()
    sns.despine(bottom=True, left=True)
    # set a 16:9 format, not in inches
    # tight layout
    plt.tight_layout()
    plt.show()