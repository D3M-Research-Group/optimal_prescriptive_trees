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


df = pd.DataFrame()

df_buffer = pd.read_csv(f'../../results/warfarin/compiled/unconstrained_agg.csv')
df_buffer = df_buffer[(df_buffer['method'] == 'Robust') & (df_buffer['depth'] == 2)][['method', 'randomization', 
                                                                                  'best_found_train']].rename(columns={'best_found_train':
                                                                                                                     'realized_outcome_train'})
    
df_buffer['method'] = 'DR (d=2) (train)'
df = pd.concat([df, df_buffer], ignore_index=True)

df_buffer = pd.read_csv(f'../../results/warfarin/compiled/unconstrained_agg.csv')
df_buffer = df_buffer[(df_buffer['method'] == 'Robust') & (df_buffer['depth'] == 2)][['method', 'randomization', 
                                                                                  'best_found_test']].rename(columns={'best_found_test':
                                                                                                                     'realized_outcome_train'})
    
df_buffer['method'] = 'DR (d=2) (test)'
df = pd.concat([df, df_buffer], ignore_index=True)

    
df_buffer = pd.read_csv(f'../../results/warfarin/compiled/policytree/raw_proba_training.csv')
for col, name in zip(['random', 'r0.06', 'r0.11'], ['0.33', 'r0.06', 'r0.11']):
    h = pd.DataFrame({'realized_outcome_train': df_buffer[col].tolist()})
    h['method'] = 'PT (train)'
    h['randomization'] = name
    h['realized_outcome_train'] *= 100
    df = pd.concat([df, h], ignore_index=False)
    
df_buffer = pd.read_csv(f'../../results/warfarin/compiled/policytree/raw_proba_old.csv')
for col, name in zip(['random', 'r0.06', 'r0.11'], ['0.33', 'r0.06', 'r0.11']):
    h = pd.DataFrame({'realized_outcome_train': df_buffer[col].tolist()})
    h['method'] = 'PT (test)'
    h['randomization'] = name
    h['realized_outcome_train'] *= 100
    df = pd.concat([df, h], ignore_index=False)

matplotlib.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), sharey=True)
for i, r in enumerate(['0.33', 'r0.06', 'r0.11']):
    
    subset = df[df['randomization'] == r]
    
    order=['DR (d=2) (train)', 'DR (d=2) (test)', 'PT (train)', 'PT (test)']
    positions = [1, 1.8, 3, 3.8]
    dic_order = {k: v for k, v in zip(order, positions)}
    sorted_dict = {key: value for key, value in sorted(dic_order.items())}
    colors = ["#E32D2D", "#FFA4A4",
          "#CCDA2F", "#E7EE92"]
    
    color_order = {k: v for k, v in zip(order, colors)}
    legend_elements = [Patch(facecolor=v, label=k, linewidth=1, edgecolor='black') for k, v in color_order.items()]
    
    sorted_dict_colors = {key: value for key, value in sorted(color_order.items())}
    
    bplot = subset.boxplot(by='method', column='realized_outcome_train',
                          positions=list(sorted_dict.values()),
                          grid=False, patch_artist=True, ax=ax[i], sym='d', return_type='dict', widths=0.8)

#     Style boxplot
#     print(bplot['realized_outcome_oos']['boxes'])
    for patch, color in zip(bplot['realized_outcome_train']['boxes'], list(sorted_dict_colors.values())):
        patch.set_facecolor(color)
        patch.set_edgecolor('0.2')
        patch.set_linewidth(1.5)
    for whisker in bplot['realized_outcome_train']['whiskers']:
        whisker.set_color('0.2')
        whisker.set_linewidth(1.5)
    for fliers in bplot['realized_outcome_train']['fliers']:
        fliers.set_markerfacecolor('0.2')
    for median in bplot['realized_outcome_train']['medians']:
        median.set_color('0.2')
        median.set_linewidth(1.5)
    for caps in bplot['realized_outcome_train']['caps']:
        caps.set_color('0.2')
        caps.set_linewidth(1.5)
    
#     xticklabels = ax[i].get_xticklabels()
#     ax[i].set_xticklabels(xticklabels, rotation = 45, ha="right")
    ax[i].axhline(y = 0.63*100, color = 'y', linestyle = '--')
    ax[i].set_xticks([])
    ax[i].set_xlabel('')
    ax[i].grid(visible=True, which='major', axis='y')
    if i == 0:
        ax[i].set_ylabel('OOSP (%)')
        ax[i].set_title('Randomized')
    else:
        ax[i].set_ylabel('')
        if i == 1:
            ax[i].set_title(r'$r = 0.06$')
            ax[i].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), prop={'size': 24}, ncol=4, loc='center')
        elif i == 2:
            ax[i].set_title(r'$r = 0.11$')
            
plt.suptitle('')

plt.subplots_adjust(wspace=0.05)
plt.savefig('figs/fig_ec3.pdf', bbox_inches='tight')
plt.show()