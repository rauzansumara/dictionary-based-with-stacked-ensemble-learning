import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import pandas as pd
import tensorflow as tf
from utils import SGCNN, Symbolic2Words
from itertools import product
from sktime.datasets import load_from_arff_to_dataframe


def run(data_name: str):

    # Datasets
    X_train, y_train = load_from_arff_to_dataframe(f'datasets/{data_name}/{data_name}_TRAIN.arff')
    X_test, y_test = load_from_arff_to_dataframe(f'datasets/{data_name}/{data_name}_TEST.arff')
    
    n_bins = [5, 7, 8, 9, 11, 12, 13, 15, 17, 18, 19, 20]
    window_size = [2, 3, 5, 7, 8]

    results = []
    for nb, ws in product(n_bins, window_size):

        s2w = Symbolic2Words(X_train, y_train, X_test, y_test, nb, ws)
        model = SGCNN(s2w[0], s2w[1], s2w[2], s2w[3])

        hist = model.fit(s2w[4], s2w[5], epochs=100, batch_size=16, 
                        validation_data=(s2w[6], s2w[7]), verbose=2, use_multiprocessing = True)

        # Evaluate the test set
        _, test_acc = model.evaluate(s2w[6], s2w[7], verbose=0)

        # save results
        results.append({"Dataset": data_name, "n_bins": nb, "window_size": ws, "Acc": test_acc})
        
    # print resuts
    # print(f'for {data_name} :')
    # print(pd.DataFrame(results)['test_acc'].groupby(['n_bins','window_size']).mean())

    # save 
    pd.DataFrame(results).to_excel(f'results/{data_name}_output.xlsx', index=False)  

    return results

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SyntaxError("Usage: python3 classify.py data_name")
    run(sys.argv[1])
