from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

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

    pets  = read_proc_time_data(keys, 'processing_time/pets-psa.txt')
    #Add column with Protocolcol name
    pets[0]['Procedure'] = 'KH-PRF-PSA.Setup'
    pets[1]['Procedure'] = 'KH-PRF-PSA.Encrypt'
    lass = read_proc_time_data(keys, 'processing_time/lass.txt')
    #Add column with protocol name
    lass[0]['Procedure'] = 'LaSS-PSA.Setup'
    lass[1]['Procedure'] = 'LaSS-PSA.Encrypt'
  
    #Create one big dataframe each for encrypt and setup
    setup = pd.concat([pets[0], lass[0]], ignore_index=True)
    encrypt = pd.concat([pets[1], lass[1]], ignore_index=True)

    #Plot graph for setup
    plot = sns.lineplot(
        data=setup, x="number", y="time", hue="Procedure",
        palette="YlGnBu_d", style='Procedure', markers=True, err_style="bars", errorbar=("se",2))

    #Set x and y labels 
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)")
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])

    plt.show()

    #Plot graph for encrypt
    plot = sns.lineplot(
        data=encrypt, x="number", y="time", hue="Procedure",
        palette="YlGnBu_d", style='Procedure', markers=True, err_style="bars", errorbar=("se",2))
    # Set x and y labels
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)")
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])
    plt.show()

def model_fit(df_tuple, protocol):
    setup = df_tuple[0]
    enc = df_tuple[1]
    
    X = setup["number"]
    y = setup["time"]
    X = sm.add_constant(X) # add y-intercept to our

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model

    # Print out the statistics
    print("=== {} Setup ===".format(protocol))
    print(model.summary())

    X = enc["number"]
    y = enc["time"]
    X = sm.add_constant(X) # add y-intercept to our

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model

    # Print out the statistics
    print("=== {} Encrypt ===".format(protocol))
    print(model.summary())

def print_values(df_tuple, protocol):
    setup = df_tuple[0]
    enc = df_tuple[1]
    
    print("=== {} Setup ===".format(protocol))
    n1024 = setup.loc[setup["number"] == 1024]["time"].mean()
    n10000 = setup.loc[setup["number"] == 10000]["time"].mean()
    print("1024: {} 10000: {}".format(n1024, n10000))
    
    print("=== {} Encrypt ===".format(protocol))
    n1024 = enc.loc[enc["number"] == 1024]["time"].mean()
    n10000 = enc.loc[enc["number"] == 10000]["time"].mean()
    print("1024: {} 10000: {}".format(n1024, n10000))

def plot_all():

    keys = [1024,2025,3025,4096,5041,6084,7056,8100,9025,10000]

    pets  = read_proc_time_data(keys, 'processing_time/pets-psa.txt')
    #Add column with Procedurecol name
    pets[0]['Procedure'] = 'KH-PRF-PSA.Setup'
    pets[1]['Procedure'] = 'KH-PRF-PSA.Encrypt'
    lass = read_proc_time_data(keys, 'processing_time/lass.txt')
    #Add column with protocol name
    lass[0]['Procedure'] = 'LaSS-PSA.Setup'
    lass[1]['Procedure'] = 'LaSS-PSA.Encrypt'
    dips = read_proc_time_data(keys,'processing_time/dipsauce.txt')
    #Add column with Procedurecol name
    dips[0]['Procedure'] = 'DIPSAUCE.Setup'
    dips[1]['Procedure'] = 'DIPSAUCE.Encrypt'
    model_fit(pets, "KH-PRF-PSA") 
    model_fit(lass, "LaSS-PSA") 
    model_fit(dips, "DIPSAUCE") 
    print_values(pets, "KH-PRF-PSA")
    print_values(lass, "LaSS-PSA") 
    print_values(dips, "DIPSAUCE") 

    setup = pd.concat([pets[0], lass[0], dips[0]], ignore_index=True)
    encrypt = pd.concat([pets[1], lass[1], dips[1]], ignore_index=True)
  
    #Plot graph for setup
    plot = sns.lineplot(
        data=setup, x="number", y="time", hue="Procedure",
        palette="YlGnBu_d", style='Procedure', markers=True, err_style="bars", errorbar=("se",2))

    #Set x and y labels 
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)")
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])
    plt.show()

    #Plot graph for encrypt
    plot = sns.lineplot(
        data=encrypt, x="number", y="time", hue="Procedure",
        palette="YlGnBu_d", style='Procedure', markers=True, err_style="bars", errorbar=("se",2))
    # Set x and y labels
    plot.set(xlabel ="Number of Users", ylabel = "Execution Time (s)")
    plot.set_xticks(keys)
    plot.set_xticklabels([str(x) for x in keys])
    plt.show()

plot_prelim() 
plot_all()

