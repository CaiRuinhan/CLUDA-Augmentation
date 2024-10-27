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
# Method = range(9)
accuracy_26_2 = [0] * len(Method)
accuracy_7_2 = [0] * len(Method)
accuracy_18_20 = [0] * len(Method)

accuracy_26_2[0] = 0.7317
accuracy_26_2[1] = 0.8049
accuracy_26_2[2] = 0.8049
accuracy_26_2[3] = 0.8537
accuracy_26_2[4] = 0.878
accuracy_26_2[5] = 0.8293
accuracy_26_2[6] = 0.8537
accuracy_26_2[7] = 0.8537
accuracy_26_2[8] = 0.8537

accuracy_7_2[0] = 0.6829
accuracy_7_2[1] = 0.7805
accuracy_7_2[2] = 0.7805
accuracy_7_2[3] = 0.7317
accuracy_7_2[4] = 0.7805
accuracy_7_2[5] = 0.7805
accuracy_7_2[6] = 0.6829
accuracy_7_2[7] = 0.7317
accuracy_7_2[8] = 0.7805

accuracy_18_20[0] = 0.7561
accuracy_18_20[1] = 0.7805
accuracy_18_20[2] = 0.7561
accuracy_18_20[3] = 0.8049
accuracy_18_20[4] = 0.7561
accuracy_18_20[5] = 0.7805
accuracy_18_20[6] = 0.7805
accuracy_18_20[7] = 0.8049
accuracy_18_20[8] = 0.7805



groups = ['(26,2)', '(7,2)', '(18,2)']
accuracy = [accuracy_26_2, accuracy_7_2, accuracy_18_20]

# Define a color palette for the bars
colors = sns.color_palette("Spectral", len(accuracy_26_2))

barWidth = 0.07
bar_positions = np.arange(len(groups))

# Create figure and adjust size
fig, ax = plt.subplots()
fig.set_size_inches([10, 8])  # adjust as needed


for i in range(len(accuracy_26_2)):
    offset = barWidth * (len(accuracy_26_2) - 1) / 2  # Calculate the offset to center the bars
    ax.bar(bar_positions - offset + i * barWidth, [group[i] for group in accuracy], width=barWidth, color=colors[i],
           edgecolor='black', linewidth=1, label=Method[i])

# Labels
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by method and pair')
ax.set_xticks(bar_positions)
ax.set_xticklabels(groups)
ax.set_ylim([0.4, 1.0])

# Place x-label in the middle of each group bar
ax.set_xlabel('(Source, Target)', ha='center')

# Legend
ax.legend(title="Methods", bbox_to_anchor=(1.02, 1), loc='upper left')

# Prevents cutting off legend or labels
fig.tight_layout()

plt.show()
