import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matplotlib.use("QtAgg")

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import KFold

from model import generate_model, generate_model_binary
from training_generator import YeastDataGenerator, YeastDataGeneratorBinary

TRAINING_DIR = "training-data"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

included_classes = ["Healthy", "Sick"]


def load_images(img_dir: str) -> (np.ndarray, np.ndarray):
    imgs = []
    labels = []
    classes = os.listdir(img_dir)
    i = 0
    for img_class in classes:
        if img_class in included_classes:  # used to filter unknown
            for img_file in os.listdir(f"{img_dir}/{img_class}"):
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
    return img_arr, label_arr


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


X, Y = load_images(TRAINING_DIR)
X, Y = unison_shuffled_copies(X, Y)
# Y[Y==2] = 0 # from healthy reject sick to cell, reject

included_classes = ["Healthy", "Sick"]
n_splits = 8
kf = KFold(n_splits=n_splits, shuffle=True)
kf_num = 0
epochs = 100
batch_size = 32

t_hist = np.zeros((n_splits, epochs))
v_hist = np.zeros((n_splits, epochs))

models = []
val_idx_list = []

for train_index, val_index in kf.split(X, Y):
    # Check whether this is a binary classification or categorical
    if len(included_classes) > 2:
        gen_class = YeastDataGenerator
    else:
        gen_class = YeastDataGeneratorBinary

    train_generator = gen_class(
        X[train_index],
        Y[train_index],
        batch_size=batch_size,
        augment=True
    )

    validation_generator = gen_class(
        X[val_index],
        Y[val_index],
        batch_size=batch_size,
        augment=False
    )

    if len(included_classes) > 2:
        model = generate_model()
    else:
        model = generate_model_binary()

    # store these so we can predict after.
    models.append(model)
    val_idx_list.append(val_index)

    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy CV={kf_num}')
    plt.savefig(f'training_{kf_num}.svg')
    model.save(f'model_end_{kf_num}.hdf5')

    t_hist[kf_num] = acc
    v_hist[kf_num] = val_acc

    kf_num += 1

epochs_range = range(epochs)

t_mean = t_hist.mean(axis=0)
t_std = t_hist.std(axis=0)
v_mean = v_hist.mean(axis=0)
v_std = v_hist.std(axis=0)

# Create the plot
plt.figure(figsize=(8, 8))
plt.plot(t_mean, label='Training accuracy')
plt.fill_between(range(len(t_mean)), t_mean - t_std, t_mean + t_std, alpha=0.2)
plt.plot(v_mean, label='Validation accuracy')
plt.fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=0.2)
plt.legend()
plt.title(f'{n_splits}-fold cross validation over {epochs} epochs')
plt.savefig(f'training_cv.svg')

for i in range(n_splits):
    np.savetxt(f"{i}_idx.txt", val_idx_list[i], fmt="%i")

preds = []
trus = []

for i in range(n_splits):
    idx = val_idx_list[i]
    pred = models[i].predict(X[idx])
    preds.append(pred)
    trus.append(Y[idx])

pred_list = []
for arr in preds:
    for pred in arr:
        pred_list.append(pred.round()[0])

tru_list = []
for arr in trus:
    for val in arr:
        tru_list.append(val)

cm = confusion_matrix(tru_list, pred_list)
tn, fp, fn, tp = cm.ravel()

sn = tp / (tp + fn)
sp = tn / (tn + fp)
acc = (tp + tn) / (tn + fp + fn + tp)
print(f"Acc: {acc}\nSen: {sn}\nSpec: {sp}")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fully fused", "Partially fused"])
disp.plot()
plt.show()
