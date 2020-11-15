import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

fp_color='g'
run_color='b'
data_file = "fixpoints_10-11-2020_11-00-46_PM_accuracy_0.001_how_many_1024_input_zero_True.pkl"

with open(data_file, 'rb') as file:
    fix_points = pickle.load(file)
print(np.mean(fix_points, axis=0).shape)
fix_points_centered = fix_points - np.mean(fix_points, axis=0)

#pca3
print('pca3')
pca3 = PCA(n_components=3)
transformed_fixpoints = pca3.fit_transform(fix_points_centered)
print(transformed_fixpoints.shape)
print(pca3.explained_variance_)

#pcaN
print('pcaN')
pcaN = PCA(n_components=50)
transformed_fixpoints_pca = pcaN.fit_transform(fix_points_centered)

tsne3=TSNE(n_components=3)
transformed_fixpoints_pca_tsne = tsne3.fit_transform(transformed_fixpoints_pca)
print(pcaN.explained_variance_)


runs_file = 'exemplary_runs_words_11-11-2020_09-01-30_PM_how_many_40.pkl'
with open(runs_file, 'rb') as file:
    words, hidden = pickle.load(file)

run_batch_size = 40
run_seqlen = 20
random_run = np.random.randint(0,40)
run_sentence = words[random_run,:]
run_hidden = hidden[random_run,:,:]
run_hidden_pca = pcaN.transform(run_hidden)
runs_fps_pcan = np.concatenate((run_hidden_pca, transformed_fixpoints_pca))
runs_fps_tsne = tsne3.fit_transform(runs_fps_pcan)
runs_tsne = runs_fps_tsne[0:run_seqlen,:]
transformed_fixpoints_pca_tsne = runs_fps_tsne[run_seqlen:,:]

print(runs_tsne.shape, transformed_fixpoints_pca_tsne.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot fix points with read out color
ax.scatter(transformed_fixpoints_pca_tsne[:,0], transformed_fixpoints_pca_tsne[:,1], transformed_fixpoints_pca_tsne[:,2], alpha=.5, c=fp_color)
#add run
ax.plot(xs=runs_tsne[:,0], ys=runs_tsne[:,1], zs=runs_tsne[:,2], c=run_color)
for i in range(run_seqlen):
    ax.text(runs_tsne[i,0], runs_tsne[i,1], runs_tsne[i,2], run_sentence[i], None)
plt.show()
