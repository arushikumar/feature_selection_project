# Default imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(data, size=11):
    plt.figure(figsize=(size,size))
    #labels = list(data.columns.values)
    sns.heatmap(data.corr())
