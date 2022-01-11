# write your code here
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data(path="mnist.npz")
(n, k, l) = x_train.shape
m = k * l
x_train = x_train.reshape(n, m)
'''
print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {x_train.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {x_train.min()}, max: {x_train.max()}")
'''
features = x_train[:6000]
targets = y_train[:6000]
my_x_train, my_x_test, my_y_train, my_y_test =\
    train_test_split(features, targets, test_size=0.3, random_state=40)
print(f"x_train shape: {my_x_train.shape}")
print(f"x_test shape: {my_x_test.shape}")
print(f"y_train shape: {my_y_train.shape}")
print(f"y_test shape: {my_y_test.shape}")

vc = pd.Series(my_y_train).value_counts(normalize=True)
print("Proportion of samples per class in train set:")
for c in np.unique(y_train):
    print(f"{c} {vc[c]}")


