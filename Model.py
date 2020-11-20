import numpy as np
import Extract as ex
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


def main(forest_path):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fire_class = ['B','C','D','E','F','G']
    data = ex.load_dataset(forest_path)
    data = ex.classify_tag(data)
    data = data.replace(fire_class,[1,2,3,4,5,6])
    x = data[['latitude','longitude']].to_numpy()
    y = data['natural'].to_numpy()

    model = KNeighborsClassifier(3)
    h = 0.02
    ax = plt.subplot()
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    plt.show()


if __name__=='__main__':
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')
