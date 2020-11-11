import numpy as np
import pandas as pd


"""create a few utility functions to help with gathering, grouping and sorting data"""

def load_dataset(csv_path):
    """
    Load the data set from the given csv path
    :param csv_path: path of file to extract
    :return: return a numpy array
    """
    # open file to read
    data = pd.read_csv(csv_path)
    # headers = ['Unnamed: 0','Unnamed: 0.1', 'fire_name','disc_clean_date','cont_clean_date','disc_date_final',
    #                       'putout_time','disc_date_pre','disc_pre_month','wstation_usaf','dstation_m','wstation_wban',
    #                       'wstation_byear','wstation_eyear','fire_mag']
    #data = data.drop(headers, axis=1)
    # load whatever data required
    return data

def obtain_total_month(data):
    """
    Takes the read csv data and returns the total amount of fires every month
    :param data: open csv file
    :return: numpy array
    """
    cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bymonths = data['discovery_month'].value_counts()
    bymonths.index = pd.CategoricalIndex(bymonths.index, categories=cats, ordered=True)
    bymonths = bymonths.sort_index()
    X = np.array([cats]).T
    Y = np.array([bymonths]).T
    months = np.hstack((X, Y))
    return months

def obtain_total_year(data):
    """
    Takes the read csv data and returns the total amount of fires every year
    :param data: open csv file
    :return: numpy array
    """
    byyears = data['disc_pre_year'].value_counts().sort_index()
    X2 = np.array([byyears.index.tolist()]).T
    Y2 = np.array([byyears]).T
    years = np.hstack((X2,Y2))
    return years

def obtain_total_cause(data):
    """
    Takes the read csv data and returns the total amount of fire causes categorized
    :param data: open csv file
    :return: numpy array
    """
    cause_cats = data['stat_cause_descr'].value_counts().sort_index()
    X = np.array([cause_cats.index.tolist()]).T
    Y = np.array([cause_cats]).T
    cause_cats = np.hstack((X,Y))
    return cause_cats

def obtain_month_year(data):
    """
    Obtains the total amount of fires each month in each year
    :param data: open csv file
    :return: numpy array
    """
    bymonthyear = data.groupby(['disc_pre_year', 'discovery_month']).discovery_month.count()
    # need a way to organize by jan-dec
    monthyear = np.array(bymonthyear.index.tolist()).T
    monthlyfire = np.array([bymonthyear]).T
    month_year = np.hstack((monthyear,monthlyfire))
    return month_year

def obtain_state(data):
    """
    Obtains the total amount of fires in each state
    :param data: open csv file
    :return: numpy array
    """
    state = data['state'].value_counts().sort_index()
    X = np.array([state.index.tolist()]).T
    Y = np.array([state]).T
    state = np.hstack((X,Y))
    return state

def obtain_cause(data, year, month):
    """
    Obtains the total causes of the month
    :param data: open csv file
    :param year: int, year
    :param month: str, month (abr. 'Jan', 'Feb' etc...)
    :return: numpy array
    """
    bycause = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    bycause = bycause['stat_cause_descr'].value_counts().sort_index()
    X = np.array([bycause.index.tolist()]).T
    Y = np.array([bycause]).T
    bycause = np.hstack((X,Y))
    return bycause

def obtain_temp(data, year, month, labels=['Temp_pre_30','Temp_pre_15','Temp_pre_7','Temp_cont']):
    """
    Gives the related temperature for each fire occurrence with the month
    :param data: open csv file
    :param year: int, year
    :param month: str, month (abr. 'Jan', 'Feb' etc...)
    :param labels: recorded temperature at either 30, 15, 7 or day of contingency
    :return: numpy array
    """
    yearmonth = ['disc_pre_year','discovery_month']
    temperature = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    temperature = temperature[labels]
    temperature = temperature.to_numpy()
    return temperature

def obtain_wind(data, year, month, labels=['Wind_pre_30','Wind_pre_15','Wind_pre_7','Wind_cont']):
    """
    Gives the related wind for each fire occurrence with the month
    :param data: open csv file
    :param year: int, year
    :param month: str, month (abr. 'Jan', 'Feb' etc...)
    :param labels: recorded wind at either 30, 15, 7 or day of contingency
    :return: numpy array
    """
    yearmonth = ['disc_pre_year', 'discovery_month']
    wind = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    wind = wind[labels]
    wind = wind.to_numpy()
    return wind

def obtain_humid(data, year, month, labels=['Hum_pre_30','Hum_pre_15','Hum_pre_7','Hum_cont']):
    """
    Gives the related humidity for each fire occurrence with the month
    :param data: open csv file
    :param year: int, year
    :param month: str, month (abr. 'Jan', 'Feb' etc...)
    :param labels: recorded humidity at either 30, 15, 7 or day of contingency
    :return: numpy array
    """
    yearmonth = ['disc_pre_year', 'discovery_month']
    humid = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    humid = humid[labels]
    humid = humid.to_numpy()
    return humid


def obtain_rain(data, year, month, labels=['Prec_pre_30','Prec_pre_15','Prec_pre_7','Prec_cont']):
    """
    Gives the related rain for each fire occurrence within the month
    :param data: open csv file
    :param year: int, year
    :param month: str, month (abr. 'Jan, 'Feb' etc...)
    :param labels: recorded rain at either 30, 15, 7 or day of contingency
    :return: numpy array
    """
    yearmonth = ['disc_pre_year', 'discovery_month']
    rain = data[(data['disc_pre_year'] == year) & (data['discovery_month'] == month)]
    labels = yearmonth + labels
    rain = rain[labels]
    rain = rain.to_numpy()
    return rain


def main(forest_path):
    data = load_dataset(forest_path)
    state = obtain_rain(data,2015,'Nov')
    print(state)

if __name__ == '__main__':
    # can use the online path or do local path
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')
    # neural networks
