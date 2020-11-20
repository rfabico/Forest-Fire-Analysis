import Extract as ex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

def graph_total_cause(data):
    data = ex.obtain_total_cause(data)
    x = data[:,0]
    y = data[:,1]
    y = y.astype(int)
    plt.xticks(rotation=90)
    plt.bar(x,y)
    plt.savefig('total_causes.png')
    plt.show()

def linear_every_month(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = ex.obtain_month_year(data)
    for entry in data:
        entry[1] = months.index(entry[1]) + 1
    data = data[:,[1,2]]
    data = data.astype(int)
    data = data[np.argsort(data[:,0])]
    x = data[:,0]
    y = data[:,1]
    x = x.reshape((-1,1))
    model = sk.LinearRegression()
    model.fit(x,y)
    y_poly_pred = model.predict(x)
    plt.scatter(x,y)
    plt.plot(x,y_poly_pred, color='red')
    plt.show()


def lat_log(data):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fire_class = ['B','C','D','E','F','G']
    data = ex.classify_tag(data)
    data = data.replace(fire_class,[1,2,3,4,5,6])
    x = data[['latitude','longitude']].to_numpy()
    y = data['natural'].to_numpy()

    model = KNeighborsClassifier(3)
    h = 0.02
    ax = plt.subplot()
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
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

def heatmap_month_year(data):
    bymonthyear = data.groupby(['disc_pre_year', 'discovery_month']).discovery_month.count()
    plot_data= bymonthyear.unstack(level=0)
    plt.figure(figsize=(15, 8))
    sb.heatmap(plot_data, linewidths=0.05, vmax=500, cmap='Oranges', fmt="1.0f", annot=True)
    plt.title('Heatmap of number of fires in states in every month in years', fontsize=15)
    plt.show()