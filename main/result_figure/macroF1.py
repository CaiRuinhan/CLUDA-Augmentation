import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Method = ['CLUDA', 
          '1w/ Time Warping',
          '2w/ Random FT + Scramble Phase',
          '3w/ Random FT + Bandstop Filter',
          '4w/ Random FT + Scramble Phase + Bandstop Filter',
          '5w/ APP-based',
          '6w/ APP-based + Pitch Shift',
          '7w/ APP-based + Bandstop Filter',
          '8w/ APP-based + Scramble Phase + Bandstop Filter']

score_26_2 = [0] * len(Method)
score_7_2 = [0] * len(Method)
score_18_20 = [0] * len(Method)

score_26_2[0] = 0.5246
score_26_2[1] = 0.5877
score_26_2[2] = 0.6574
score_26_2[3] = 0.6849
score_26_2[4] = 0.6789
score_26_2[5] = 0.6603
score_26_2[6] = 0.6821
score_26_2[7] = 0.6609
score_26_2[8] = 0.6779

score_7_2[0] = 0.5427
score_7_2[1] = 0.5632
score_7_2[2] = 0.5923
score_7_2[3] = 0.5066
score_7_2[4] = 0.5603
score_7_2[5] = 0.5520
score_7_2[6] = 0.5090
score_7_2[7] = 0.5495
score_7_2[8] = 0.5603

score_18_20[0] = 0.5879
score_18_20[1] = 0.6747
score_18_20[2] = 0.6212
score_18_20[3] = 0.5923
score_18_20[4] = 0.5982
score_18_20[5] = 0.5521
score_18_20[6] = 0.5421
score_18_20[7] = 0.5325
score_18_20[8] = 0.6631



groups = ['(26,2)', '(7,2)', '(18,2)']
score = [score_26_2, score_7_2, score_18_20]

# Define a color palette for the bars
colors = sns.color_palette("Spectral", len(score_26_2))

barWidth = 0.07
bar_positions = np.arange(len(groups))

# Create figure and adjust size
fig, ax = plt.subplots()
fig.set_size_inches([10, 8])  # adjust as needed


for i in range(len(score_26_2)):
    offset = barWidth * (len(score_26_2) - 1) / 2  # Calculate the offset to center the bars
    ax.bar(bar_positions - offset + i * barWidth, [group[i] for group in score], width=barWidth, color=colors[i],
           edgecolor='black', linewidth=1, label=Method[i])

# Labels
ax.set_ylabel('Macro F1 score')
ax.set_title('F1 score by method and pair')
ax.set_xticks(bar_positions)
ax.set_xticklabels(groups)
ax.set_ylim([0.2, 0.8])

# Place x-label in the middle of each group bar
ax.set_xlabel('(Source, Target)', ha='center')

# Legend
ax.legend(title="Methods", bbox_to_anchor=(1.02, 1), loc='upper left')

# Prevents cutting off legend or labels
fig.tight_layout()

plt.show()
