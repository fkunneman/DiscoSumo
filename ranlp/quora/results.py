__author__='thiagocastroferreira'

import os
import json

from sklearn.metrics import f1_score, accuracy_score

if __name__ == '__main__':
    path = 'results'
    for fname in os.listdir(path):
        if fname != 'dev':
            results = json.load(open(os.path.join(path, fname)))
            print(fname)
            y_real = [w['y_real'] for w in results]
            y_pred = [w['y_pred'] for w in results]
            print('Accuracy: ', round(accuracy_score(y_real, y_pred), 2))
            print('F1-Score: ', round(f1_score(y_real, y_pred), 2))
            print(10 *'-')
