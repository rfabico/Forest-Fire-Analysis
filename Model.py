import numpy as np
import Extract as ex
import functions as fs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.linear_model as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import glob


def main(forest_path):
    data = ex.load_dataset(forest_path)
    show = fs.year_pred_poly(data)

if __name__ == '__main__':
    main(forest_path='https://raw.githubusercontent.com/rfabico/Forest-Fire-Analysis/main/FW_Veg_Rem_Combined.csv')