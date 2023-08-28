import numpy as np
from math import factorial
from matplotlib import pyplot as plt
import argparse
import yaml

def compare_results(args):
    with open(f"Comparisons/{args.comparison_list}.yaml", "r") as f:
        data = yaml.safe_load(f)
    results_dirs = data['results_dirs']
    plt_title = data['plt_title']
    x_label = data['x_label']
    y_label = data['y_label']
    column = data.get('column', 1)
    scores = {}
    ci = {}
    steps = []
    colors = ['b', 'g', 'r', 'orange', 'purple', 'grey', 'black']
    for i, (label, results_paths) in enumerate(results_dirs.items()):
        results = []
        for result_dir in results_paths:
            results.append(np.loadtxt(f"{result_dir}/results.csv", delimiter=','))
        results = np.stack(results, axis=2)
        scores[label] = np.mean(results, axis=2)
        ci[label] = 1.95 * np.std(results, axis = 2)/np.sqrt(results.shape[2])
        
        indices = range(0,len(scores[label][:, 0]),2)
        plt.plot(scores[label][indices, 0], scores[label][indices,column], label=label, color=colors[i])
        plt.fill_between(scores[label][indices, 0], (scores[label][indices,column]-ci[label][indices,column]), (scores[label][indices,column]+ci[label][indices,column]), color=colors[i], alpha=.1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plt_title)
        
    plt.legend()
    plt.savefig('Figure.png')
    plt.show()