import numpy as np
import Extract as ex
import pandas as pd
import sklearn.linear_model as sk

class Forest:
    """

    """

    def __init__(self,X,Y,theta=None):
        self.X = X
        self.Y = Y
        self.theta = theta
        model = sk.LogisticRegression()

    def predict(self,X):
        return self.model.predict(X)

    def fit(self, X, Y):
        self.theta = self.model.fit(X,Y)


def main(forest_path):
    data = ex.load_dataset(forest_path)
    sum_state = data.groupby(['state']).count().sort_index()
    print(sum_state)


if __name__ == '__main__':
    # can use the online path or do local path
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')
