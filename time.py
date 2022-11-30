from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import pylab
import pandas as pd
from termcolor import colored
import csv
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

#reads data from processing and radio files

def read_proc_time_data(keys,fi):
    data = pd.read_csv(fi)
    #convert from ticks to seconds
    data['time'] = data['time'].apply(lambda x: x/65536.0)

    #sort encrypt and setup by removing rows.
    setup = data[~data['type'].isin(['e'])]
    encrypt = data[~data['type'].isin(['s'])]
    
    return (setup, encrypt)

def plot_prelim():

    keys = [1024,2025,3025,4096,5041,6084,7056,8100,9025,10000]

    pets  = read_proc_time_data(keys, 'processing_time/pets-psa-dummy.txt')
    #Add column with Protocolcol name
    pets[0]['Protocol'] = 'Pets-PSA'
    pets[1]['Protocol'] = 'Pets-PSA'
    lass = read_proc_time_data(keys, 'processing_time/lass-dummy.txt')
    #Add column with protocol name
    lass[0]['Protocol'] = 'LaSS'
    lass[1]['Protocol'] = 'LaSS'
  
    #Create one big dataframe each for encrypt and setup
    setup = pd.concat([pets[0], lass[0]], ignore_index=True)
    encrypt = pd.concat([pets[1], lass[1]], ignore_index=True)

    #Plot graph for setup
    plot = sns.lineplot(
        data=setup, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", style='Protocol', markers=True, err_style="bars", errorbar=("se",2))

    #Set x and y labels 
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)", title ='Setup')
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])

    plt.show()

    #Plot graph for encrypt
    plot = sns.lineplot(
        data=encrypt, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", style='Protocol', markers=True, err_style="bars", errorbar=("se",2))
    # Set x and y labels
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)", title ='Encrypt')
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])
    plt.show()


def plot_all():

    keys = [1024,2025,3025,4096,5041,6084,7056,8100,9025,10000]

    pets  = read_proc_time_data(keys, 'processing_time/pets-psa-dummy.txt')
    #Add column with Protocolcol name
    pets[0]['Protocol'] = 'Pets-PSA'
    pets[1]['Protocol'] = 'Pets-PSA'
    lass = read_proc_time_data(keys, 'processing_time/lass-dummy.txt')
    #Add column with protocol name
    lass[0]['Protocol'] = 'LaSS'
    lass[1]['Protocol'] = 'LaSS'
    dips = read_proc_time_data(keys,'processing_time/dipsauce-dummy.txt')
    #Add column with Protocolcol name
    dips[0]['Protocol'] = 'DIPSAUCE'
    dips[1]['Protocol'] = 'DIPSAUCE'
  
    setup = pd.concat([pets[0], lass[0], dips[0]], ignore_index=True)
    encrypt = pd.concat([pets[1], lass[1], dips[1]], ignore_index=True)
  
    #Plot graph for setup
    plot = sns.lineplot(
        data=setup, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", style='Protocol', markers=True, err_style="bars", errorbar=("se",2))

    #Set x and y labels 
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)", title ='Setup')
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])
    plt.show()

    #Plot graph for encrypt
    plot = sns.lineplot(
        data=encrypt, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", style='Protocol', markers=True, err_style="bars", errorbar=("se",2))
    # Set x and y labels
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)", title ='Encrypt')
    plt.show()
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])

plot_prelim() 
plot_all()

