# write your code here

import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
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


(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data(path="mnist.npz")
(n, k, l) = x_train.shape
m = k * l
x_train = x_train.reshape(n, m)

features = x_train[:6000]
targets = y_train[:6000]
my_x_train, my_x_test, my_y_train, my_y_test =\
    train_test_split(features, targets, test_size=0.3, random_state=40)

models = [KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40),
          RandomForestClassifier(random_state=40)
]

normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(my_x_train)
x_test_norm = normalizer.transform(my_x_test)
models_scores = {}


param_grid_knc = dict(n_neighbors=[3, 4],
                      weights=['uniform', 'distance'],
                      algorithm=['auto', 'brute'])
gs_knc = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid_knc,
                      scoring='accuracy', n_jobs=-1)
gs_knc.fit(x_train_norm, my_y_train)
y_pred = gs_knc.best_estimator_.predict(x_test_norm)
score = accuracy_score(my_y_test, y_pred)

print('K-nearest neighbours algorithm')
#print(gs_knc.best_params_)
print(f'best estimator: {gs_knc.best_estimator_}')
print(f'accuracy: {score}')

param_grid_rfc = dict(n_estimators = [300, 500],
                      max_features = ['auto', 'log2'],
                      class_weight = ['balanced', 'balanced_subsample'])
gs_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=40),
                      param_grid=param_grid_rfc,
                      scoring='accuracy', n_jobs=-1)
gs_rfc.fit(x_train_norm, my_y_train)
y_pred = gs_rfc.best_estimator_.predict(x_test_norm)
score = accuracy_score(my_y_test, y_pred)


print('Random forest algorithm')
#print(gs_knc.best_params_)
print(f'best estimator: {gs_rfc.best_estimator_}')
print(f'accuracy: {score}')
