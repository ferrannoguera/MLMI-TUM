import argparse
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('ggplot')

def plot(runs):

    mn = min([e for k,v in runs.items() for e in v])
    mx = max([e for k,v in runs.items() for e in v])
    offset = 0.05
    
    for i,r in enumerate(runs.keys()):
        runs[r] = [*runs[r], sum(runs[r])/len(runs[r])]
        if i == 0:
            plt.vlines([i + 0.5 for i in range(len(runs[r])-1)], mn-offset, mx+offset, color="gray")
        plt.scatter([i for i in range(len(runs[r]))], runs[r], label=r.split("(")[0])

    plt.rc('legend', fontsize="large")
    plt.legend()
    plt.ylabel("Accuracy", size="large")
    cls = list(range(len(runs[list(runs.keys())[0]])))
    plt.xticks(ticks=cls, labels =[f"class_{i + 1}" if i != len(cls) - 1 else "class_avg" for i in cls], size="large")
    
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='visualization of per class accuracies')
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()

    with open(args.path, "r") as f:
        runs = json.load(f)
    plot(runs)