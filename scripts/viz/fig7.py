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

results_synthetic = pd.read_csv('../../results/synthetic/compiled/DR_new.csv')
results_synthetic['budget'] *= 100
results_synthetic['oos_optimal_treatment'] *= 100

matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for pred, ml, x, y in zip(['tree', 'tree', 'log', 'log'], ['linear', 'lasso', 'linear', 'lasso'], [0, 0, 1, 1], [0, 1, 0, 1]):
    sub = results_synthetic[(results_synthetic['propensity_score_pred'] == pred) & \
                                (results_synthetic['ml'] == ml)][['budget', 'randomization', 
                                            'oos_optimal_treatment']].groupby(['budget', 'randomization']).agg('mean').reset_index()
    sns.lineplot(data=sub, 
         y='oos_optimal_treatment', x='budget', hue='randomization', ax=ax[x, y])
    
    if x != 0 or y != 1:
        leg = ax[x, y].get_legend()
        leg.remove()
    else:
        ax[x, y].legend(loc=(1.1, -0.5))
    
    if x == 0:
        ax[x, y].set_xlabel('')
    else:
#         ax[x, y].set_xlabel('Probability of Correct\nTreatment Assignment in Data')
        ax[x, y].set_xlabel('Budget (%)')
        
    if y == 0:
        ax[x, y].set_ylabel('OOSP (%)')
    ax[x, y].grid(True)
#     ax[x, y].set_xticks([0.1, 0.25, 0.5, 0.75, 0.9])
fig.text(0.32, 0.83, 'LR', ha='center', va='center', weight='bold', size='20')
fig.text(0.72, 0.83, 'Lasso', ha='center', va='center', weight='bold', size='20')
fig.text(0.03, 0.64, 'DT', ha='center', va='center', weight='bold', size='20')
fig.text(0.03, 0.28, 'Log', ha='center', va='center', weight='bold', size='20')
fig.text(0.93, 0.65, 'Probability of\nCorrect Treatment\nAssignment in Data', ha='left', va='center', size='17')
plt.subplots_adjust(top=0.8, wspace=0.05)
plt.savefig('figs/fig7.pdf', bbox_inches='tight')
plt.show()