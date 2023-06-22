import os
from random import sample

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from model import generate_model_binary

matplotlib.use("QtAgg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

included_classes = ["Healthy", "Sick"]

def load_images(img_dir: str) -> (np.ndarray, np.ndarray):
    imgs = []
    labels = []
    filenames = []
    classes = os.listdir(img_dir)
    i = 0
    for img_class in classes:
        if img_class in included_classes:  # used to filter unknown
            for img_file in sample(os.listdir(f"{img_dir}/{img_class}"), 500):
                img = io.imread(f"{img_dir}/{img_class}/{img_file}")  # Loaded as ZXY
                img = np.swapaxes(img, 0, 1)  # to XZY
                img = np.swapaxes(img, 1, 2)  # to XYZ
                img = resize(img, (64, 64, 2))
                if "nikon" in img_file:
                    img *= (65535 / 2047)  # max pixel val at 16bit and saturation threshold
                elif "leica" in img_file:
                    img *= (65535 / 7500)  # highest observed value

                imgs.append(img)
                labels.append(i)
            i += 1  # go to next valid class
    img_arr = np.array(imgs).reshape((len(imgs), 64, 64, 2))
    label_arr = np.array(labels).reshape((len(labels)))
    return img_arr, label_arr, filenames


model = generate_model_binary()
model.load_weights(f'model_end_0.hdf5')  # random model
classes = ["Healthy", "Sick"]

X, Y, filenames = load_images("training-data")

inter_model = tf.keras.Model(model.input, model.get_layer("dense").output)
feats = inter_model(X).numpy()
np.savetxt("feats.csv", feats, delimiter=",")
np.savetxt("labels.csv", np.array(Y), fmt="%d")
with open("filenames.csv", "a") as fp:
    fp.writelines("\n".join(filenames))

class_names = ["Fully fused", "Partially fused"]
tsne = TSNE(perplexity=300,n_iter=2000).fit_transform(feats)
groups = np.unique(Y)
fig, ax = plt.subplots(1,1, figsize=(15,15))
for label in groups:
    i = np.where(Y==label)
    ax.scatter(tsne[i,0],tsne[i,1], label=class_names[label])
ax.legend()
ax.set_title("t-SNE of features extracted from the last fully-connected layer")
plt.show()


pca = PCA(n_components=3).fit_transform(feats)
groups = np.unique(Y)
fig, ax = plt.subplots(1,1, figsize=(15,15))
for label in groups:
    i = np.where(Y==label)
    ax.scatter(pca[i,0],pca[i,1], label=class_names[label])
ax.legend()
ax.set_title("PCA of features extracted from the last fully-connected layer")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.show()


fig = plt.figure(figsize=(14,9))
ax = fig.add_subplot(111, 
                     projection='3d')
 
for label in groups:
    i = np.where(Y==label)
    ax.scatter(pca[i,0],pca[i,1],pca[i,2], label=class_names[label])
 
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)
 
ax.view_init(30, 125)
ax.legend()
plt.title("3D PCA plot")
plt.show()


svd = TruncatedSVD(n_components=3).fit_transform(feats)
groups = np.unique(Y)
fig, ax = plt.subplots(1,1, figsize=(15,15))
for label in groups:
    i = np.where(Y==label)
    ax.scatter(svd[i,0],svd[i,1], label=class_names[label])
ax.legend()
ax.set_title("tSVD of features extracted from the last fully-connected layer")
ax.set_xlabel("C1")
ax.set_ylabel("C2")
plt.show()