# write your code here

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    target_pred = model.predict(features_test)
    score = accuracy_score(target_test, target_pred)
    print(f'Model: {model}\nAccuracy: {score}\n')
    return score
''
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

models = [KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40),
          RandomForestClassifier(random_state=40)
]
'''
best_model = None
best_score = 0
for m in models:
    score = fit_predict_eval(
        model=m,
        features_train=my_x_train,
        features_test=my_x_test,
        target_train=my_y_train,
        target_test=my_y_test
    )
    if score > best_score:
        best_model = m
        best_score = score

n = str(best_model).find('(')
str_best_model = str(best_model)[:n]
print(f'Without normalization: {str_best_model} - {round(best_score, 3)}\n')
'''

normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(my_x_train)
x_test_norm = normalizer.transform(my_x_test)
models_scores = {}
for m in models:
    v = fit_predict_eval(
        model=m,
        features_train=x_train_norm,
        features_test=x_test_norm,
        target_train=my_y_train,
        target_test=my_y_test
    )
    n = str(m).find('(')
    k = str(m)[:n]
    models_scores[k] = v
print(f'The answer to the 1st question: yes\n')
sorted_scores = sorted(models_scores.items(), key=lambda x: x[1], reverse=True)
answer2 = f'The answer to the 2nd question: '
for (k, v) in sorted_scores[:2]:
    answer2 += f'{k}-{round(v, 3)}, '
answer2 = answer2[:-2]
print(answer2)

