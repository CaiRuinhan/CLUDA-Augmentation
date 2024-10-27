import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os

src = 18
tar = 20
# Load the embeddings from the saved .npz file
embeddings_save_folder = f'experiment_WISDM_{src}_{tar}'
embeddings_save_path = f'experiment_WISDM_{src}_{tar}/embedding_{src}_{tar}.npz'
data = np.load(embeddings_save_path)
embeddings = data['embeddings_trg']
labels = data['y_trg']



# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)


# Assuming labels are integers starting from 0
unique_labels = np.unique(labels)
colors = cm.get_cmap('tab10', len(unique_labels))  # 'tab10' is a good discrete colormap

plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap=colors)
plt.title('t-SNE Visualization of Embeddings')
plt.colorbar()

# Save the figure to the embeddings_save_folder
save_file_name = f'embedding_{src}_{tar}_tsne.png'
save_path = os.path.join(embeddings_save_folder, save_file_name)
plt.savefig(save_path)

print("t-SNE visualization saved to:", save_path)
