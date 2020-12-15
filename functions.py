import Extract as ex
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


# what we improved on
def linear_every_month(data):
    """
    Linear regression
    :param data:
    :return:
    """
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


def year_poly_reg(data):
    data = ex.obtain_total_year(data)
    data = np.delete(data,0,0)
    x = data[:,0]
    x = x.reshape((-1,1))
    y = data[:,1]
    model = make_pipeline(PolynomialFeatures(5), sk.LinearRegression())
    model.fit(x,y)
    pred_y = model.predict(x)
    plt.scatter(x,y)
    plt.plot(x,pred_y, color='red')
    plt.show()


def year_pred_poly(data):
    data = ex.obtain_total_year(data)
    data = np.delete(data,0,0)
    x = data[:,0]
    x = x.reshape((-1,1))
    y = data[:,1]
    model = make_pipeline(PolynomialFeatures(5), sk.Ridge(alpha=0.1, fit_intercept=False ))
    model.fit(x,y)
    ridge = model.named_steps['ridge']
    pred_y = model.predict(x)
    for i in range(1,5):
        year = 2015+i
        value = model.predict([[year]])
        x = np.vstack([x,[year]])
        pred_y = np.append(pred_y,value)
        y = np.append(y,value)
    plt.scatter(x,y)
    plt.plot(x,pred_y, color='red')
    plt.show()

# what we used & finalized
def heatmap_month_year(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bymonthyear = data.groupby(['disc_pre_year', 'discovery_month']).discovery_month.count()
    plot_data = bymonthyear.unstack(level=0)
    plot_data = plot_data.reindex(months, axis=0)
    plt.figure(figsize=(15, 8))
    sb.heatmap(plot_data, linewidths=0.05, vmax=500, cmap='Oranges', fmt="1.0f", annot=True)
    plt.title('Heatmap of number of fires in states in every month in years', fontsize=15)
    plt.show()



def month_year_poly(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = ex.obtain_month_year(data)
    for entry in data:
        entry[1] = months.index(entry[1]) + 1
    data = data.astype(int)
    data = data[np.lexsort((data[:,1],data[:,0]))]
    data = np.delete(data,0,0)
    x = data[:,[0,1]]
    y = data[:,2]
    model = make_pipeline(PolynomialFeatures(5), sk.Ridge(alpha=0.001, fit_intercept=False))
    model.fit(x, y)
    y_poly_pred = model.predict(x)
    temp = np.zeros((x[:,0].size,1))
    for i in range(0,x[:,0].size):
        temp[i] = i + 1
    plt.scatter(temp,y)
    plt.plot(temp, y_poly_pred, color='red')
    plt.title("Regression of fires every month each year")
    plt.show()

def month_year_pred(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = ex.obtain_month_year(data)
    for entry in data:
        entry[1] = months.index(entry[1]) + 1
    data = data.astype(int)
    data = data[np.lexsort((data[:,1],data[:,0]))]
    data = np.delete(data,0,0)
    x = data[:,[0,1]]
    y = data[:,2]
    model = make_pipeline(PolynomialFeatures(5), sk.Ridge(alpha=0.001, fit_intercept=False))
    model.fit(x, y)

    y_poly_pred = model.predict(x)
    for i in range(1,13):
        year = 2016
        month = i
        value = model.predict([[year,month]])
        x = np.vstack([x, [year,month]])
        y_poly_pred = np.append(y_poly_pred,value)
        y = np.append(y,value)
    temp = np.zeros((x[:,0].size,1))
    for i in range(0,x[:,0].size):
        temp[i] = i + 1

    plt.scatter(temp,y)
    plt.plot(temp, y_poly_pred, color='red')
    plt.title("Regression of fires every month each year")
    plt.show()


def obtain_month_states(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    state = ex.obtain_state(data)
    state_list = state[:,0]
    fire_list = []
    for st in state_list:
        state_fires = []
        print(st)
        for i in range (1992,2016):
            for k in months:
                month_num = months.index(k) + 1
                result = ex.obtain_monthly_state(data,st,i,k)
                result = result['discovery_month'].value_counts().sort_index()
                if result.empty:
                    result = 0
                    fire = [i, month_num, result]
                    state_fires.append(fire)
                else:
                    X = np.array([i, month_num])
                    X = X.astype(int)
                    Y = np.array([result])
                    Y = Y.astype(int)
                    Y = Y.squeeze()
                    fire = np.hstack((X, Y)).tolist()
                    state_fires.append(fire)
        fire_list.append(state_fires)
    return fire_list


def save_csv(forest_path):
    data = ex.load_dataset(forest_path)
    data = obtain_month_states(data) # fs
    state = ex.load_dataset(forest_path)
    state = ex.obtain_state(state)
    state_list = state[:,0]
    state_list = state_list.tolist()
    for i in state_list:
        index = state_list.index(i)
        values = data[index]
        np.savetxt("state_fire/" + i + "_total" + ".csv", values, delimiter=',', fmt='%d')
    return


def test(csv_path):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    total = 0
    for file in glob.glob('state_fire/*.csv'):
        data = pd.read_csv(file,header=None)
        data = data.to_numpy()
        x = data[:,[0,1]]
        y = data[:,2]
        model = make_pipeline(PolynomialFeatures(5), sk.Ridge(alpha=0.001, fit_intercept=False))
        model.fit(x, y)
        y_poly_pred = model.predict(x)
        value = model.predict([[2016,1]])
        print(value)
        value = value[0]
        total += value
    print("total value is: " + total.astype(str))


def last_twelve(data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = ex.obtain_month_year(data)
    for entry in data:
        entry[1] = months.index(entry[1]) + 1
    data = data.astype(int)
    data = data[np.lexsort((data[:, 1], data[:, 0]))]
    data = np.delete(data, 0, 0)
    two_fifteen = data[-12:]
    x1 = data[:, [0, 1]]
    y1 = data[:, 2]
    x = x1[:-12]
    y = y1[:-12]
    model = make_pipeline(PolynomialFeatures(5), sk.Ridge(alpha=0.001, fit_intercept=False))
    model.fit(x, y)
    y_poly_pred = model.predict(x)
    for i in range(1,13):
        values = model.predict([[2015,i]])
        x = np.vstack([x,[2015,i]])
        y_poly_pred = np.append(y_poly_pred,values)
        y = np.append(y,values)
    temp = list(range(1,y_poly_pred.size + 1))
    print(two_fifteen)
    print(y[-12:])
    plt.scatter(temp,y)
    plt.plot(temp, y_poly_pred, color='red')
    plt.title("Regression of fires every month each year - prediction of last 12")
    plt.show()
    
    
# What we didn't use - scratched or did not fit what we needed
def graph_total_cause(data):
    """
    Graph to show use the total causes of
    :param data:
    :return:
    """
    data = ex.obtain_total_cause(data)
    x = data[:,0]
    y = data[:,1]
    y = y.astype(int)
    plt.xticks(rotation=90)
    plt.bar(x,y)
    plt.savefig('total_causes.png')
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

