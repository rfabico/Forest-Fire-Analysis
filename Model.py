import numpy as np
import Extract as ex
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

def create_poly(k, X):
    """
    Generates a polynomial feature map using the data x.
    The polynomial map should have powers from 0 to k
    Output should be a numpy array whose shape is (n_examples, k+1)

    Args:
        X: Training example inputs. Shape (n_examples, 2).
    """
    for n in range(len(X[0]), len(X[0]) + (k - 1)):
        toAdd = np.power(X[:, 0], n)
        toAdd = toAdd.reshape((len(toAdd)), 1)
        X = np.hstack((X, toAdd))
    return X


def main(forest_path):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    data = ex.load_dataset(forest_path)
    data = ex.obtain_rain(data,2015,'Feb')
    x = data[:,[2,3]]
    y = data[:,4]
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    model = sk.LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    plt.scatter(x_test[:,0], y_test, c='red')
    plt.scatter(x_test[:,1], y_test, c='blue')
    plt.plot(x_test,y_pred)
    plt.savefig('test.png')
    plt.show()

    # create the model

if __name__ == '__main__':
    # can use the online path or do local path
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')
