import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

# RnC
# CF + CT
# policytree
# Our method (IPW, DM, DR)
# KPT, BPT -- need to reconfigure this
# add dumb predictor (assign everything 1)
df = pd.DataFrame()
y = 'oos_optimal_treatment'
for m, m_method in zip(['IPW', 'DM', 'DR', 'Kallus', 'Bertsimas'], ['IPW', 'DM', 'DR', 'K-PT', 'B-PT']):
    df_buffer = pd.read_csv(f'../../results/synthetic/compiled/{m}.csv')
    df_buffer['method'] = m_method
    if m == 'DR':
        df_buffer = df_buffer[(df_buffer['budget'] == 1.0) & (df_buffer['propensity_score_pred'] == 'tree') &\
                             (df_buffer['ml'] == 'linear')]
    elif m == 'DM':
        df_buffer = df_buffer[(df_buffer['budget'] == 1.0) & \
                             (df_buffer['ml'] == 'linear')]
    elif m == 'IPW':
        df_buffer = df_buffer[(df_buffer['propensity_score_pred'] == 'tree')]
    
    df = pd.concat([df, df_buffer[['method', 'randomization', y]]], ignore_index=True)

# causal forest and trees
for m, m_name in zip(['cf', 'ct'], ['CF', 'CT']):
    df_buffer = pd.read_csv(f'../../results/synthetic/compiled/CF/{m}_raw.csv')
#     df_trans = pd.DataFrame(columns=['method', 'randomization', 'realized_outcome_oos'])
    cols=[f'p{i}' for i in [0.1, 0.25, 0.5, 0.75, 0.9]]
    for col in cols:
        h = pd.DataFrame({y: df_buffer[col].tolist()})
        h['method'] = m_name
        h['randomization'] = float(col[1:])
        df = pd.concat([df, h], ignore_index=False)
        
df_buffer = pd.read_csv(f'../../results/synthetic/compiled/RC/raw.csv')
df_buffer = df_buffer.rename(columns={'oosp': y, 'prob_opt': 'randomization'})
df_buffer = df_buffer[df_buffer['method'].isin(['lr'])][['method', 'randomization', y]]
df_buffer['method'] = 'RC (LR)'
df = pd.concat([df, df_buffer], ignore_index=False)

df = df.groupby(['method', 'randomization']).agg('mean').reset_index()
df['randomization'] = df['randomization'].astype(str)

# RnC
# CF + CT
# policytree
# Our method (IPW, DM, DR)
# KPT, BPT -- need to reconfigure this
# add dumb predictor (assign everything 1)
df1 = pd.DataFrame()
for m, m_method in zip(['DR', 'Kallus', 'Bertsimas'], ['DR', 'K-PT', 'B-PT']):
    df_buffer = pd.read_csv(f'../../results/synthetic/compiled/{m}.csv')
    df_buffer['method'] = m_method
    if m == 'DR':
        df_buffer = df_buffer[(df_buffer['budget'] == 1.0)]
        df_buffer = df_buffer[~((df_buffer['ml'] == 'lasso') & (df_buffer['propensity_score_pred'] == 'log'))]
        df_buffer['ml'] = df_buffer['ml'].map({'linear': 'LR', 'lasso': 'Lasso'})
        df_buffer['propensity_score_pred'] = df_buffer['propensity_score_pred'].map({'log': 'Log', 'tree': 'DT'})
        df_buffer['method'] = df_buffer.apply(lambda row: f"{row['method']} ({row['ml']}, {row['propensity_score_pred']})", axis=1)
    elif m == 'DM':
        df_buffer = df_buffer[(df_buffer['budget'] == 1.0) & \
                             (df_buffer['ml'] == 'linear')]
    elif m == 'IPW':
        df_buffer = df_buffer[(df_buffer['propensity_score_pred'] == 'tree')]
    
    df1 = pd.concat([df1, df_buffer[['method', 'randomization', y]]], ignore_index=True)
    
# policytree
df_buffer = pd.read_csv(f'../../results/synthetic/compiled/policytree/raw.csv')
cols=[f'p{i}' for i in [0.1, 0.25, 0.5, 0.75, 0.9]]
for col in cols:
    h = pd.DataFrame({y: df_buffer[col].tolist()})
    h['method'] = 'PT'
    h['randomization'] = float(col[1:])
    df1 = pd.concat([df1, h], ignore_index=False)
    
df1 = df1.groupby(['method', 'randomization']).agg('mean').reset_index()
df1['randomization'] = df1['randomization'].astype(str)
    
df[y] = df[y]*100
df1[y] = df1[y]*100

colors = ["#FFC7C7", "#FFA4A4", "#EB6262",
         "#BFBFBF", "#878787",
          "#CCDA2F",
         "#7B79CF"]

customPalette = sns.color_palette(colors)


colors1 = ["#EB6262", "#EB6262", "#EB6262",
         "#BFBFBF", "#878787"]

customPalette1 = sns.color_palette(colors1)

dash_list = [(4, 1.5), (1, 1)]

colors = ["#FFC7C7", "#FFA4A4", "#EB6262",
         "#BFBFBF", "#878787",
         "#7B79CF"]

customPalette = sns.color_palette(colors)


colors1 = ["#EB6262", "#EB6262", "#EB6262", "#CCDA2F"]

customPalette1 = sns.color_palette(colors1)

import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8), sharey=True)
for i in range(2):
    if i == 0:
        include = ['IPW', 'DM', 'DR', 'K-PT', 'B-PT', 'RC (LR)']
        sns.lineplot(data=df[df['method'].isin(include)], x='randomization', y=y, hue='method', 
                hue_order = include,
                marker='o', markersize=12, ax=ax[i], linewidth=3, palette=customPalette)
        ax[i].axhline(y = 0.486046*100, color = 'r', linestyle = '--')
        ax[i].set_ylabel('OOSP (%)')
        
        order_dummy=['IPW (DT)', 'DM (LR)', 'DR (LR, DT)', 'K-PT', 'B-PT', 'R&C (LR)']
        color_order_legend = {k: v for k, v in zip(order_dummy, colors)}
        legend_elements = [Line2D([0], [0], color=v, label=k, linewidth=3, marker='o', markersize=10) for k, v in color_order_legend.items()]
        
        ax[i].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=22)
    elif i == 1:
        include = ['DR (LR, DT)', 'DR (Lasso, DT)', 'DR (LR, Log)', 'PT']
        
        sns.lineplot(data=df1[df1['method'].isin(include)], x='randomization', y=y, hue='method', style='method',
            hue_order = include, markers=['o', 'X', '^', 'o'], markersize=12, ax=ax[i], linewidth=3, palette=customPalette1,
                    dashes=['', (4, 1.5), (1, 1), ''])
        ax[i].axhline(y = 0.486046*100, color = 'r', linestyle = '--')
        
        order_dummy=['DR (LR, DT)', 'DR (Lasso, DT)', 'DR (LR, Log)', 'PT', '1', '2']
        color_dummy=["#EB6262", "#EB6262", "#EB6262", "#CCDA2F", "#CCDA2F", "#CCDA2F"]
        marker_dummy=['o', '^', 'X', 'o', 'o', 'o']
        color_order_legend = {k: (v, v1) for k, v, v1 in zip(order_dummy, color_dummy, marker_dummy)}
        legend_elements = [Line2D([0], [0], color=v[0], label=k, linewidth=3, marker=v[1], markersize=10) for k, v in color_order_legend.items()]
        # ax[i].legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=22)
        
        handles, labels = ax[i].get_legend_handles_labels()
        # copy the handles
        handles = [copy.copy(ha) for ha in handles]
        # set the linewidths to the copies
        [ha.set_linewidth(3) for ha in handles ]
        [ha.set_markersize(10) for ha in handles ]
        ax[i].legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=22)

    ax[i].set_xlabel('Probability of Correct\nTreatment Assignment in Data')
    ax[i].grid(visible=True, axis='y')
    
plt.subplots_adjust(wspace=0.05)
plt.savefig('figs/fig3.pdf', bbox_inches='tight')
plt.show()