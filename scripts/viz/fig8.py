import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
import pickle
import warnings
warnings.filterwarnings("ignore")


warfarin_outcome = pd.read_csv('../../results/warfarin/compiled/DM.csv')

matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))

for i, r in enumerate(['0.33', 'r0.06', 'r0.11']):
    subset = warfarin_outcome[warfarin_outcome['randomization'] == r]
    sns.boxplot(data=subset, y='realized_disparity', x='fairness', ax=ax[i], palette="Greens")
#     ax[i].set_title(f'{r}')
    if i != 0:
        ax[i].set_ylabel('')
    else:
        ax[i].set_ylabel('Realized Disparity (W-NW)')
    ax[i].set_xlabel(r'Fairness Constraint $\delta$')
    ax[i].axhline(y = 0, color = 'y', linestyle = '--')
    ax[i].set_xticklabels([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1.0], rotation = 45, ha="right")

plt.subplots_adjust(wspace=0.2)
plt.savefig('figs/fig8.pdf', bbox_inches='tight')
plt.show()