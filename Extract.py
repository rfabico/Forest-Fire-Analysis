import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


"""create a few utility functions to help with gathering, grouping and sorting data"""

def load_dataset(csv_path):
    """
    Load the data set from the given csv path
    :param csv_path: path of file to extract
    :param labal_col_y: data to
    :return: return a numpy array
    """
    # open file to read
    data = pd.read_csv(csv_path)
    # create category of months to use as a way to organize the data
    cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bymonths = data['discovery_month'].value_counts()
    bymonths.index = pd.CategoricalIndex(bymonths.index, categories=cats, ordered=True)
    bymonths = bymonths.sort_index()

    # insert into numpy array
    X = np.array(cats)
    Y = np.array(bymonths)
    months = np.vstack((X,Y))

    # organize data by year now
    byyears = data['disc_pre_year'].value_counts().sort_index()
    X2 = byyears.index.tolist()
    Y2 = np.array(byyears)
    years = np.vstack((X2,Y2))

    # need a way to organize by month,year now
    bymonthyear = data.groupby(['disc_pre_year','discovery_month']).discovery_month.count()
    pair = bymonthyear.index.tolist()
    # test to check amount
    filter = data[(data['disc_pre_year'] == 2015) & (data['discovery_month'] == 'Nov')]
    print(filter.head(100))

def main(forest_path):
    load_dataset(forest_path)

if __name__ == '__main__':
    # can use the online path or do local path
    main(forest_path='https://raw.githubusercontent.com/varunr89/smokey/master/FW_Veg_Rem_Combined.csv')