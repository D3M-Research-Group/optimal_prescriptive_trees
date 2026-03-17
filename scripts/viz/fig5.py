import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle
import warnings
warnings.filterwarnings("ignore")

# RnC, + RnC on our most tailored RnC
# CF + CT
# policytree
# Our method (IPW, DM, DR)
# KPT, BPT -- need to reconfigure this
# add dumb predictor (assign everything majority class)

# open our methods first
df = pd.DataFrame()
for m in ['IPW', 'DM', 'DR']:
    df_buffer = pd.read_csv(f'../../results/warfarin/compiled/{m}.csv')
    df_buffer = df_buffer[df_buffer['fairness'] == 1.0][['method', 'randomization', 'realized_outcome_oos']]
    
    if m == 'DR':
        m = 'DR (d=2)'
        df_buffer['method'] = m
    df = pd.concat([df, df_buffer], ignore_index=True)

    
# df = pd.concat([df, pd.DataFrame({'method': ['DR (d=2)'], 'Age': [0], 'Gender': ['M']}
             
# DR (d=3)
df_buffer = pd.read_csv(f'../../results/warfarin/compiled/unconstrained_agg.csv')
df_buffer = df_buffer[(df_buffer['method'] == 'Robust') & (df_buffer['depth'] == 3)][['method', 'randomization', 
                                                                                      'best_found_test']].rename(columns={'best_found_test':
                                                                                                                         'realized_outcome_oos'})
df_buffer['method'] = 'DR (d=3)'
df_buffer['realized_outcome_oos'] /= 100
df = pd.concat([df, df_buffer], ignore_index=True)

# DR (d=4)
df_buffer = pd.read_csv(f'../../results/warfarin/compiled/unconstrained_agg.csv')
df_buffer = df_buffer[(df_buffer['method'] == 'Robust') & (df_buffer['depth'] == 4)][['method', 'randomization', 
                                                                                      'best_found_test']].rename(columns={'best_found_test':
                                                                                                                         'realized_outcome_oos'})
df_buffer['method'] = 'DR (d=4)'
df_buffer['realized_outcome_oos'] /= 100
df = pd.concat([df, df_buffer], ignore_index=True)

for m, m_method in zip(['Kallus', 'Bertsimas'], ['K-PT', 'B-PT']):
    df_buffer = pd.read_csv(f'../../results/warfarin/compiled/{m}.csv')
    df_buffer = df_buffer[['method', 'randomization', 'realized_outcome_oos']]
    df_buffer['method'] = m_method
    df = pd.concat([df, df_buffer], ignore_index=True)
    
# causal forest and trees
for m, m_name in zip(['cf', 'cf_untuned', 'ct'], ['CF', 'CF (untuned)', 'CT']):
    df_buffer = pd.read_csv(f'../../results/warfarin/compiled/CF/{m}_baseline_raw.csv')
#     df_trans = pd.DataFrame(columns=['method', 'randomization', 'realized_outcome_oos'])
    for col, name in zip(['random', 'r0.06', 'r0.11'], ['0.33', 'r0.06', 'r0.11']):
        h = pd.DataFrame({'realized_outcome_oos': df_buffer[col].tolist()})
        h['method'] = m_name
        h['randomization'] = name
        df = pd.concat([df, h], ignore_index=False)

# policytree
df_buffer = pd.read_csv(f'../../results/warfarin/compiled/policytree/raw_proba_old.csv')
for col, name in zip(['random', 'r0.06', 'r0.11'], ['0.33', 'r0.06', 'r0.11']):
    h = pd.DataFrame({'realized_outcome_oos': df_buffer[col].tolist()})
    h['method'] = 'PT'
    h['randomization'] = name
    df = pd.concat([df, h], ignore_index=False)

df_buffer = pd.read_csv(f'../../results/warfarin/compiled/RC/rc_raw.csv')
df_buffer_random = df_buffer[df_buffer['randomization'] == '0.33']
df_buffer_random1 = df_buffer_random[df_buffer_random['model'] == 'balanced_rf']
df_buffer_random1['model'] = 'best'
df_buffer_random = pd.concat([df_buffer_random[df_buffer_random['model'] != 'lrrf'], df_buffer_random1], ignore_index=True)
df_buffer_random['model'] = df_buffer_random['model'].map({'balanced_rf': 'R&C (RF)', 'best': 'R&C (Best)',
                                                          'balanced_lr': 'R&C (Log)'})

df_buffer_other = df_buffer[df_buffer['randomization'] != '0.33']
df_buffer_other['model'] = df_buffer_other['model'].map({'balanced_rf': 'R&C (RF)', 'lrrf': 'R&C (Best)',
                                                          'balanced_lr': 'R&C (Log)'})

df_buffer = pd.concat([df_buffer_random, df_buffer_other], ignore_index=True).rename(columns={'model': 'method',
                                                                                              'oosp': 'realized_outcome_oos'})

df = pd.concat([df, df_buffer[['method', 'randomization', 'realized_outcome_oos']]], ignore_index=False)

df['realized_outcome_oos'] = df['realized_outcome_oos']*100

colors = ["#FFC7C7", "#FFA4A4", "#EB6262", "#E32D2D", "#C81C1C",
         "#BFBFBF", "#878787",
         "#9CCF9B", "#67D165", "#21C51D",
          "#CCDA2F",
         "#B1B0CD", "#7B79CF", "#4543C0"]
# Set your custom color palette
customPalette = sns.color_palette(colors)

matplotlib.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), sharey=True)
for i, r in enumerate(['0.33', 'r0.06', 'r0.11']):
    
    subset = df[df['randomization'] == r]
    
    order=['IPW', 'DM', 'DR (d=2)', 'DR (d=3)', 'DR (d=4)',
                                'K-PT', 'B-PT',
                                'CF', 'CF (untuned)', 'CT', 'PT',
                                'R&C (Best)', 'R&C (Log)', 'R&C (RF)']
    positions = [1, 1.6, 2.2, 2.8, 3.4, 5, 5.6, 7.2, 7.8, 8.4, 10, 11.6, 12.2, 12.8]
    dic_order = {k: v for k, v in zip(order, positions)}
    sorted_dict = {key: value for key, value in sorted(dic_order.items())}
    colors = ["#FFC7C7", "#FFA4A4", "#EB6262", "#E32D2D", "#C81C1C",
         "#BFBFBF", "#878787",
         "#9CEAB6", "#67D165", "#21C51D",
          "#CCDA2F",
         "#B1B0CD", "#7B79CF", "#4543C0"]
    
    color_order = {k: v for k, v in zip(order, colors)}
    
    order_dummy=['IPW', 'DM', r'DR ($d=2$)', r'DR ($d=3$)', r'DR ($d=4$)',
                                'K-PT', 'B-PT', '1', '2', '3',
                                'CF', 'CF (untuned)', 'CT', 'PT', '4',
                                'R&C (Best)', 'R&C (Log)', 'R&C (RF)', '5', '6']
    colors_dummy = ["#FFC7C7", "#FFA4A4", "#EB6262", "#E32D2D", "#C81C1C",
         "#BFBFBF", "#878787", "#878787", "#878787", "#878787",
         "#9CEAB6", "#67D165", "#21C51D",
          "#CCDA2F", "#878787",
         "#B1B0CD", "#7B79CF", "#4543C0", "#878787", "#878787"]
    color_order_legend = {k: v for k, v in zip(order_dummy, colors_dummy)}
    legend_elements = [Patch(facecolor=v, label=k, linewidth=1, edgecolor='black') for k, v in color_order_legend.items()]
    
    sorted_dict_colors = {key: value for key, value in sorted(color_order.items())}
    
    bplot = subset.boxplot(by='method', column='realized_outcome_oos',
                          positions=list(sorted_dict.values()),
                          grid=False, patch_artist=True, ax=ax[i], sym='d', return_type='dict', widths=0.6)

#     Style boxplot
#     print(bplot['realized_outcome_oos']['boxes'])
    for patch, color in zip(bplot['realized_outcome_oos']['boxes'], list(sorted_dict_colors.values())):
        patch.set_facecolor(color)
        patch.set_edgecolor('0.2')
        patch.set_linewidth(1.5)
    for whisker in bplot['realized_outcome_oos']['whiskers']:
        whisker.set_color('0.2')
        whisker.set_linewidth(1.5)
    for fliers in bplot['realized_outcome_oos']['fliers']:
        fliers.set_markerfacecolor('0.2')
    for median in bplot['realized_outcome_oos']['medians']:
        median.set_color('0.2')
        median.set_linewidth(1.5)
    for caps in bplot['realized_outcome_oos']['caps']:
        caps.set_color('0.2')
        caps.set_linewidth(1.5)
    
#     xticklabels = ax[i].get_xticklabels()
#     ax[i].set_xticklabels(xticklabels, rotation = 45, ha="right")
    ax[i].axhline(y = 0.63*100, color = 'y', linestyle = '--')
    ax[i].set_xticks([])
    ax[i].set_xlabel('')
    ax[i].grid(visible=True, which='major', axis='y')
    ax[i].set_title('')
    if i == 0:
        ax[i].set_ylabel('OOSP (%)')
#         ax[i].set_title('Randomized')
    else:
        ax[i].set_ylabel('')
        if i == 1:
#             ax[i].set_title(r'$r = 0.06$')
            ax[i].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.22), prop={'size': 19}, ncol=4, loc='center')
#         elif i == 2:
#             ax[i].set_title(r'$r = 0.11$')
            
plt.suptitle('')

plt.subplots_adjust(wspace=0.05)
plt.savefig('figs/fig5.pdf', bbox_inches='tight')
plt.show()