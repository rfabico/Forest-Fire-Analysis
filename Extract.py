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
    bymonths.plot(kind='bar', fontsize=14, color='r')
    plt.show()
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
    # test to check amount & can use to filter our certain conditions of the month
    temperature = obtain_temp(data,2015,'Nov')
    print(temperature.head(100))
    #filter = data[(data['disc_pre_year'] == 2015) & (data['discovery_month'] == 'Nov')]
    #filter = filter[['disc_pre_year','discovery_month','Temp_pre_30']]
    #print(filter.head)



def obtain_temp(data, year, month, labels=['Temp_pre_30','Temp_pre_15','Temp_pre_7','Temp_cont']):
    yearmonth = ['disc_pre_year','discovery_month']
    temperature = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    temperature = temperature[labels]
    return temperature

def obtain_wind(data, year, month, labels=['Wind_pre_30 ','Wind_pre_15','Wind_pre_7','Wind_cont']):
    yearmonth = ['disc_pre_year', 'discovery_month']
    wind = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    wind = wind[labels]
    return wind


def obtain_humid(data, year, month, labels=['Hum_pre_30','Hum_pre_15','Hum_pre_7','Hum_cont']):
    yearmonth = ['disc_pre_year', 'discovery_month']
    humid = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    humid = humid[labels]
    return humid

def obtain_rain(data, year, month, labels=['Prec_pre_30','Prec_pre_15','Prec_pre_7','Prec_cont']):
    yearmonth = ['disc_pre_year', 'discovery_month']
    rain = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    rain = rain[labels]
    return rain

def main(forest_path):
    load_dataset(forest_path)

if __name__ == '__main__':
    # can use the online path or do local path
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')
