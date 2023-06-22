import json
import os

import cv2
import numpy as np
import tensorflow as tf
from skimage import io
from skimage.transform import resize

from model import generate_model_binary

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

included_classes = ["Healthy", "Sick", "Reject"]

def load_images(acq_dir: str) -> (np.ndarray, np.ndarray):
    imgs = []
    for img_dir in os.listdir(acq_dir):
        if os.path.isdir(f'{acq_dir}/{img_dir}'):
            for img_file in os.listdir(f'{acq_dir}/{img_dir}'):
                img = io.imread(f"{acq_dir}/{img_dir}/{img_file}")  # Loaded as ZXY
                img = np.swapaxes(img, 0, 1)  # to XZY
                img = np.swapaxes(img, 1, 2)  # to XYZ
                img = resize(img, (64, 64, 2))
                if "nikon" in img_file:
                    img *= (65535 / 2047)  # max pixel val at 16bit and saturation threshold
                elif "leica" in img_file:
                    img *= (65535 / 7500)  # highest observed value

                imgs.append(img)
    img_arr = np.array(imgs).reshape((len(imgs), 64, 64, 2))
    return img_arr


def load_models(path):
    models = []
    for i in range(8):
        model = generate_model_binary()
        model.load_weights(f'{path}/model_end_{i}.hdf5')
        models.append(model)
    return models

models_reject = load_models("model-6-7-cell-reject")
models_type = load_models("model-6-12-healthy-sick")
classes = ["Healthy", "Reject", "Sick"]

for acq_dir in os.listdir("rois"):
    x = load_images(f"rois/{acq_dir}")

    results = []
    for model in models_reject:
        results.append(model.predict(x))

    results_type = []
    for model in models_type:
        results_type.append(model.predict(x))

    mean_res = np.mean(np.array(results),axis=0)
    mean_res_type = np.mean(np.array(results_type),axis=0)

    class_count = {
        "Healthy": 0,
        "Reject": 0,
        "Sick": 0,
    }

    for i, res in enumerate(mean_res):
        arg = res.round()
        if arg == 0: # cell
            type_arg = mean_res_type[i].round()
            if type_arg == 0:
                class_count["Healthy"] += 1
            else:
                class_count["Sick"] += 1
        else:
            class_count["Reject"] += 1

    print("-"*20)
    print(f"{acq_dir}")
    print(class_count)
    print("-" * 20)
    with open("pred-counts/"+acq_dir + ".json", 'w') as fp:
        json.dump(class_count, fp)