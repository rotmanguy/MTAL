import math

import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import seaborn as sns

writing_dir = "plots/cross_domain"

if not os.path.isdir(writing_dir):
    os.makedirs(writing_dir)

tasks = ["dp", "ner"]

for task in tasks:
    df = pd.read_csv(os.path.join(writing_dir, "cross_domain_" + task + ".csv"))
    df.set_index('Model', inplace=True)
    df = df.transpose()
    min_val = df.min().min()
    max_val = df.max().max()
    min_max_val = math.ceil(max(-1 * min_val, max_val))
    fig, ax = plt.subplots(figsize=(8, 8))
    # color map
    # cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    cmap = "PiYG" #"bwr"  #"YlGnBu"
    # plot heatmap
    sns.heatmap(df, annot=True, fmt=".2f",
               linewidths=5, cmap=cmap,
               cbar_kws={"shrink": .35}, square=True,
                vmin=-min_max_val, vmax=min_max_val)
    # ticks
    yticks = [i for i in df.index]
    xticks = [i for i in df.columns]

    plt.yticks(plt.yticks()[0], labels=yticks, horizontalalignment='right')
    plt.xticks(plt.xticks()[0], labels=xticks, horizontalalignment='right')
    # title
    plt.xlabel("")
    plt.ylabel("")
    plt.title("", fontsize=18)
    #plt.show()
    plt.savefig(os.path.join(writing_dir, 'cross_domain_heatmap_' + task + '.png'))
    plt.close()