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
    #sort encrypt and setup by removing rows.
    setup = data[~data['type'].isin(['e'])]
    encrypt = data[~data['type'].isin(['s'])]
    
    return (setup, encrypt)

def plot_setup():

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
  
    sns.lineplot(
        data=setup, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", err_style="bars", errorbar=("se",2))
   
    plt.show()

def plot_encrypt():

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
  
    sns.lineplot(
        data=encrypt, x="number", y="time", hue="Protocol",
        palette="YlGnBu_d", err_style="bars", errorbar=("se",2))
   
    plt.show()

plot_setup() 
plot_encrypt()

