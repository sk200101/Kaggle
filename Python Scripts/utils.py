from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
import pandas as pd


def optimize_and_evaluate_model(pipeline, parameter_grid, method_name, X, Y):
    gridsearch = GridSearchCV(pipeline, parameter_grid, cv=5, scoring='neg_mean_squared_error')
    gridsearch.fit(X, Y)
    print(gridsearch.best_estimator_)
    print("\nMethod is " + method_name)
    print "Root Mean Square Error", (np.sqrt(abs(gridsearch.best_score_)))
    print "\n#################################################\n"
    return


def read_data(file_path):
    # Read data
    df_train = pd.read_csv(file_path + 'train.csv')
    df_test = pd.read_csv(file_path + 'test.csv')
    return df_train, df_test


def output_file(ids, saleprices):
    # write classifications to file
    print("writing data to file")
    myfile = open("../data/submission.csv", 'wb')
    wr = csv.writer(myfile)
    # header
    wr.writerow(["Id", "SalePrice"])
    # data rows
    for id, saleprice in zip(ids, saleprices):
        wr.writerow([id, saleprice])

    myfile.close()
    return
