import numpy as np
import pickle
import os
import scipy.io
import copy

adv_robust_root_path = os.path.join("..", "adv_robust")

data_root_path = os.path.join(adv_robust_root_path, "tf_autozoom_2d_randVec20")
data_path = os.path.join(data_root_path, "all_results.pkl")
with open(data_path, "rb") as f:
    d = pickle.load(f)
true_train_x = d["true_x"]
true_train_y = d["true_y"]
adv_train_x = d["adv_x"]


# read in adversarial results for testing data
data_path = os.path.join(data_root_path, "all_test_results.pkl")
with open(data_path, "rb") as f:
    d = pickle.load(f)
true_test_x = d["true_x"]
true_test_y = d["true_y"]
adv_test_x = d["adv_x"]


def transform_data(d, img_dim=32):
    n = d.shape[1]
    return np.transpose(copy.deepcopy(d)).reshape([n, img_dim, img_dim, 1])


scipy.io.savemat("face_data.mat", {"true_train_x": transform_data(true_train_x), 
                                   "true_train_y": true_train_y,
                                   "true_test_x" : transform_data(true_test_x),
                                   "true_test_y" : true_test_y,
                                   "adv_train_x" : transform_data(adv_train_x),
                                   "adv_test_x"  : transform_data(adv_test_x)})