from utils import *
from models import *
from feature_engineering import *


def feature_enginnering(df_train, df_test):

    # Remove huge basement outlier from training data
    df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 4000].index)

    # Split train into x and y
    df_train_y = df_train.loc[:, ['SalePrice']]
    df_train_x = df_train.drop(['SalePrice'], 1)

    # Feature Engineering
    df_train_y = transform_target(df_train_y)  # train y
    df_train_x = transform_features(df_train_x)  # train x
    df_test_x = transform_features(df_test)  # test x

    # Encode Categorical Variables and put into sci-kit friendly format
    df_train_x = pd.get_dummies(df_train_x, drop_first=False, dummy_na=False)
    df_train_x = df_train_x.fillna(0)
    df_test_x = pd.get_dummies(df_test_x, drop_first=False, dummy_na=False)
    df_test_x = df_test_x.fillna(0)

    # Ensure test and training have the same variables
    test_cols = set(df_test_x.columns.values)
    train_cols = set(df_train_x.columns.values)
    # Remove cols from train that are not in test
    df_train_x = df_train_x.drop(list(train_cols - test_cols), 1)
    # Remove cols from train that are not in test
    df_test_x = df_test_x.drop(list(test_cols - train_cols), 1)

    # Double check that all columns are the same in test and train
    for test_col, train_col in zip(list(df_test_x.columns.values), list(df_train_x.columns.values)):
        if test_col != train_col:
            print test_col, train_col, 'not the same'

    return df_train_x, df_train_y, df_test_x



def main():
    file_path = '../data/'
    df_train, df_test=read_data(file_path)
    df_train_x, df_train_y, df_test_x = feature_enginnering(df_train, df_test)

    train_x = df_train_x.as_matrix()
    train_y = df_train_y.as_matrix().ravel()
    test_x = df_test_x.as_matrix()

    # Fit lasso regression
    pipeline_lasso = lasso(train_x, train_y, df_train_x)
    # Fit XGBoosting
    pipeline_xg = XGBoosting(train_x,train_y, df_train_x)
    # Fit random forest
    random_forest(train_x,train_y, df_train_x,df_train_x)
    # Fit PCA
    PCA(train_x,train_y)
    # Fit emsemble model of lasso and XGboosting
    ensemble(train_x, train_y,pipeline_lasso,pipeline_xg)

    ########### Predict #############
    lasso_prediction = pipeline_lasso.fit(train_x, train_y).predict(test_x)
    xg_prediction = pipeline_xg.fit(train_x, train_y).predict(test_x)
    ensemble_prediction = lasso_prediction * 0.75 + xg_prediction * 0.25

    ids = list(df_test['Id'].values)
    saleprices = list(np.exp(ensemble_prediction))
    output_file(ids, saleprices)

    return







