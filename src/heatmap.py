import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

drug='chlorpromazine'
model_nums = ['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']

length=20000
grid1 = []
grid1_lu = []
for m in model_nums:
    output_folder = f'outputs/{drug}/model_{m}_{drug}'
    output_folder_lu = f'outputs/{drug}/model_{m}_{drug}_kemp'
    # get fitted drug-binding parameters
    drug_fit_score = []
    drug_fit_score_lu = []
    for j, m in enumerate(model_nums):
        df = pd.read_csv(f'{output_folder}/fits/{model_nums[j]}_fit_{length}_points.csv')
        df_lu = pd.read_csv(f'{output_folder_lu}/fits/{model_nums[j]}_fit_{length}_points.csv')
        drug_fit_score.append(max(df['score']))
        drug_fit_score_lu.append(max(df_lu['score']))
    grid1.append(drug_fit_score)
    grid1_lu.append(drug_fit_score_lu)

length=40000
grid2 = []
grid2_lu = []
for m in model_nums:
    output_folder = f'outputs/{drug}/model_{m}_{drug}/opt_synth_data'
    output_folder_lu = f'outputs/{drug}/model_{m}_{drug}_kemp/opt_synth_data'
    # get fitted drug-binding parameters
    drug_fit_score = []
    drug_fit_score_lu = []
    for j, m in enumerate(model_nums):
        df = pd.read_csv(f'{output_folder}/fits/{model_nums[j]}_fit_{length}_points.csv')
        df_lu = pd.read_csv(f'{output_folder_lu}/fits/{model_nums[j]}_fit_{length}_points.csv')
        drug_fit_score.append(max(df['score']))
        drug_fit_score_lu.append(max(df_lu['score']))
    grid2.append(drug_fit_score)
    grid2_lu.append(drug_fit_score_lu)

data_array1 = np.array(grid1)
data_array2 = np.array(grid2)
data_array3 = np.array(grid1_lu)
data_array4 = np.array(grid2_lu)

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace = 0.2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])

vmax = max(max(max(grid1[:])), max(max(grid2[:])))
vmin = min(min(min(grid1[:])), min(min(grid2[:])))

vmax_lu = max(max(max(grid1_lu[:])), max(max(grid2_lu[:])))
vmin_lu = min(min(min(grid1_lu[:])), min(min(grid2_lu[:])))

#vmin1 = min(min(grid1[:]))
vmax1 = max(max(grid1[:]))
vmax1_lu = max(max(grid1_lu[:]))

#vmin2 = min(min(grid2[:]))
vmax2 = max(max(grid2[:]))
vmax2_lu = max(max(grid2_lu[:]))

# Create the heatmap using imshow
cax1 = ax1.imshow(data_array1, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
cax2 = ax2.imshow(data_array2, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
cax3 = ax3.imshow(data_array3, cmap='viridis', interpolation='nearest', vmin=vmin_lu, vmax=vmax_lu)
cax4 = ax4.imshow(data_array4, cmap='viridis', interpolation='nearest', vmin=vmin_lu, vmax=vmax_lu)

# Add a color bar to the side of the heatmap
cb1 = fig.colorbar(cax1)
cb2 = fig.colorbar(cax2)
cb3 = fig.colorbar(cax3)
cb4 = fig.colorbar(cax4)

cb1.ax.tick_params(labelsize=10)
cb2.ax.tick_params(labelsize=10)
cb3.ax.tick_params(labelsize=10)
cb4.ax.tick_params(labelsize=10)

# Find and outline the highest value in each row
for i, row in enumerate(data_array1):
    max_value = np.max(row)
    threshold1 = max_value - 10**4
    threshold2 = max_value - 10**5
    for j, value in enumerate(row):
        if value >= threshold1:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='green', facecolor='none')
            ax1.add_patch(rect)
        elif value >= threshold2:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)

# Find and outline the highest value in each row
for i, row in enumerate(data_array2):
    max_value = np.max(row)
    threshold1 = max_value - 10**4
    threshold2 = max_value - 10**5
    for j, value in enumerate(row):
        if value >= threshold1:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
        elif value >= threshold2:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)

# Find and outline the highest value in each row
for i, row in enumerate(data_array3):
    max_value = np.max(row)
    threshold1 = max_value - 10**4
    threshold2 = max_value - 10**5
    for j, value in enumerate(row):
        if value >= threshold1:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='green', facecolor='none')
            ax3.add_patch(rect)
        elif value >= threshold2:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)

# Find and outline the highest value in each row
for i, row in enumerate(data_array4):
    max_value = np.max(row)
    threshold1 = max_value - 10**4
    threshold2 = max_value - 10**5
    for j, value in enumerate(row):
        if value >= threshold1:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='green', facecolor='none')
            ax4.add_patch(rect)
        elif value >= threshold2:
            rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, linewidth=1.5, edgecolor='red', facecolor='none')
            ax4.add_patch(rect)

for i in np.arange(0, len(model_nums)):
    ax1.scatter(i,i,marker='.',color='k',s=10)
    ax2.scatter(i,i,marker='.',color='k',s=10)
    ax3.scatter(i,i,marker='.',color='k',s=10)
    ax4.scatter(i,i,marker='.',color='k',s=10)

# Add labels (optional)
ax1.set_ylabel('Data-Generating Model', fontsize = 10)
ax1.set_title('Milnes', fontsize = 10)

# Add labels (optional)
ax2.set_title('Optimised protocol', fontsize = 10)

ax3.set_xlabel('Fitted Model', fontsize = 10)
ax3.set_ylabel('Data-Generating Model', fontsize = 10)
ax3.set_title('Milnes', fontsize = 10)

# Add labels (optional)
ax4.set_xlabel('Fitted Model', fontsize = 10)
#ax2.set_ylabel('Data-Generating Model', fontsize = 10)
ax4.set_title('Optimised protocol', fontsize = 10)

# Set ticks
ax1.set_xticks(np.arange(0,len(model_nums),1),model_nums)
ax1.set_yticks(np.arange(0,len(model_nums),1),model_nums)

# Set ticks
ax2.set_xticks(np.arange(0,len(model_nums),1),model_nums)
ax2.set_yticks(np.arange(0,len(model_nums),1),model_nums)

# Set ticks
ax3.set_xticks(np.arange(0,len(model_nums),1),model_nums)
ax3.set_yticks(np.arange(0,len(model_nums),1),model_nums)

# Set ticks
ax4.set_xticks(np.arange(0,len(model_nums),1),model_nums)
ax4.set_yticks(np.arange(0,len(model_nums),1),model_nums)

ax1.tick_params(axis='both', which='major', labelsize=8.5)
ax1.tick_params(axis='both', which='minor', labelsize=8.5)
ax2.tick_params(axis='both', which='major', labelsize=8.5)
ax2.tick_params(axis='both', which='minor', labelsize=8.5)
ax3.tick_params(axis='both', which='major', labelsize=8.5)
ax3.tick_params(axis='both', which='minor', labelsize=8.5)
ax4.tick_params(axis='both', which='major', labelsize=8.5)
ax4.tick_params(axis='both', which='minor', labelsize=8.5)

# Get the current xtick labels
labels = ax1.get_xticklabels()

# Adjust the position of the label
labels[12].set_position((labels[12].get_position()[0], labels[12].get_position()[1] - 0.05))
labels[14].set_position((labels[14].get_position()[0], labels[14].get_position()[1] - 0.05))

# Get the current xtick labels
labels = ax2.get_xticklabels()

# Adjust the position of the label
labels[12].set_position((labels[12].get_position()[0], labels[12].get_position()[1] - 0.05))
labels[14].set_position((labels[14].get_position()[0], labels[14].get_position()[1] - 0.05))

# Get the current xtick labels
labels = ax3.get_xticklabels()

# Adjust the position of the label
labels[12].set_position((labels[12].get_position()[0], labels[12].get_position()[1] - 0.05))
labels[14].set_position((labels[14].get_position()[0], labels[14].get_position()[1] - 0.05))

# Get the current xtick labels
labels = ax4.get_xticklabels()

# Adjust the position of the label
labels[12].set_position((labels[12].get_position()[0], labels[12].get_position()[1] - 0.05))
labels[14].set_position((labels[14].get_position()[0], labels[14].get_position()[1] - 0.05))

fig.text(0.5, 0.93, 'Chlorpromazine, no hERG model discrepancy', ha='center', fontsize=10)
fig.text(0.5, 0.48, 'Chlorpromazine, hERG model discrepancy', ha='center', fontsize=10)

# Show the plot
plt.savefig(f'figures/heatmap_{drug}_both_new.png', dpi=1200, bbox_inches='tight')
