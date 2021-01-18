import random

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import decimal
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sys
from os.path import dirname
from pathlib import Path
from pmlb import fetch_data

root = dirname(dirname(__file__))
print(f'Root code folder: {root}')
sys.path.append(root)

from soft.utils.math_space import create_nonlinear_math_space, create_hard_math_space
from soft.utils.gp_classifier import GpClassifier

seed = 2
result_path = '/tmp'
random.seed(seed)

datasets = [
    'prnn_crabs', 'heart_h', 'crx',
    'haberman', 'breast', 'flare',
    'pima', 'german', 'heart_c',
    'credit_g', 'buggyCrx', 'prnn_synth'
]
for dataset_name in datasets:
    dataset = fetch_data(dataset_name, local_cache_dir = './data/').to_numpy()
    result_dir = f'{result_path}/{dataset_name}'
    Path(result_dir).mkdir(parents = True, exist_ok = True)

    num_rows, num_cols = dataset.shape

    models = [
        ('Decision Tree', DecisionTreeClassifier(max_depth = 5)),
        ('KNeighbors', KNeighborsClassifier()),
        ('SVC', SVC(gamma = 2, C = 1)),
        ('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))),
        ('Random Forest', RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)),
        ('MLP', MLPClassifier(alpha = 1, max_iter = 1000)),
        ('ADA', AdaBoostClassifier()),
        ('Gaussian NB', GaussianNB()),
        ('SGP Classifier',
         GpClassifier(create_nonlinear_math_space(num_cols - 1), max_generations = 100, population_size = 64,
                      with_operator_mutation = True)),
        ('GP Classifier',
         GpClassifier(create_hard_math_space(num_cols - 1), max_generations = 100, population_size = 64,
                      with_operator_mutation = False)),
    ]

    results = {}
    for i in range(20):

        print('=======================================')
        print(f'Run: {i+1}')

        np.random.shuffle(dataset)

        if dataset_name == 'titanic':
            y = dataset[:, num_cols - 1] - min(dataset[:, num_cols - 1])
            y = y / 2
        elif dataset_name == 'postoperative_patient_data':
            y = dataset[:, num_cols - 1]
            y = y / 2
        else:
            y = dataset[:, num_cols - 1] - min(dataset[:, num_cols - 1])

        class_balance = round(np.mean(y), 2)
        features = num_cols
        instances = num_rows
        min_class = np.min(y)
        max_class = np.max(y)

        file_object = open(f'{result_dir}/_info.txt', 'a')
        file_object.write(f'class balance: {class_balance}\n')
        file_object.write(f'features: {features}\n')
        file_object.write(f'instances: {instances}\n')
        file_object.write(f'min: {min_class}\n')
        file_object.write(f'max: {max_class}\n')
        file_object.close()

        X = dataset[:, 0:num_cols - 1]

        X = X / X.max(axis = 0)
        X = X[:, ~np.isnan(X).any(axis = 0)]
        X = X.astype(decimal.Decimal)
        y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

        for name, model in models:

            if name not in results:
                results[name] = []

            print('====')
            print(f'Testing {name} on {dataset_name}')

            if name in ['GP Soft', 'GP Hard']:
                model.set_test(X_test, y_test)

            model.fit(X_train, y_train)
            train_score = balanced_accuracy_score(y_train, model.predict(X_train))
            test_score = balanced_accuracy_score(y_test, model.predict(X_test))

            print(f'Train Score: {train_score}')
            print(f'Test Score: {test_score}')

            results[name].append(test_score)

            file_object = open(f'{result_dir}/{name}.txt', 'a')
            file_object.write(f'{test_score} - {train_score}\n')
            file_object.close()

        fig = plt.figure(figsize = (10, 10))
        fig.suptitle(str.upper(dataset_name))
        ax = fig.add_subplot(111)
        box = plt.boxplot(results.values(), patch_artist = True)

        colors = ['tan'] * (len(models) - 1)
        colors.append('pink')
        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor = color)

        ax.set_xticklabels(results.keys(), rotation = 'vertical')
        plt.savefig(f'{result_dir}/report_{i}.png')
