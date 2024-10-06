#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob as glob
import numpy as np
from IPython.display import display
import time
#from Common_Constants import *  
#import Common_Constants as my
#import importlib
#importlib.reload(my)
from Common_Functions import *
#from Common_Constants2 import * 
#from Common_Constants2_full_CCF_export import * 
from Common_Constants2_2024_Aug5_edit import *

'''Standard Data QC Protocol is to:

    Run qc_clean_data_loader for each folder. 
    Then run qc_year_intro_analysis on each folder. 
    Run flatlinef for all nox in met. 
    If flatlines check them out with flatline_lookup.
    Run wind_nancombo_check on met and wind columns. 
    
    Do notebook_quick_graphs for species specific investigation after turning on matplotlib notebook for zoom capabilities 






'''



    
  





    
def flatlinef(df, column_name, nr=4):
    
    '''
    Find nox flatlines. Does not account for NaN interactions. 
    
    Arguments:
        df: dataframe
        column_name: Which nox column to do.
        nr: Min number of repeated values to trigger. 
        Currently nr=5 means there are 6 occurences in a row.
    
    Returns four variables. a, b, c, d = flatlinef(df, column_name, nr=5)
        naughty_list: the values in a list
        first_date: date of first occurence of value in a list
        last_date: date of last occurence of value, in a list 
        list(pairs): all previous packaged in truple list.   
    
    '''
    
    
    full_list = list(df[column_name].values)
    naughty_list = []
    first_date = []
    last_date = []
    
    for index in range(len(full_list)):
        current_value = full_list[index]
        last_seen_date = df.at[index, "time"]  # keep track of last seen date

        if current_value in naughty_list:
            last_seen_date = df.at[index, "time"]  # update last seen date
            continue 
        else:
            counter = 0
            start_index = index  # keep track of start index
            start_date = df.at[index, "time"]  # keep track of start date
            
            for n in range(index+1, len(full_list)):  # inner loop to check for repeated values
                if (full_list[n] == current_value):
                    counter += 1
                    last_seen_date = df.at[n, "time"]  # update last seen date
                else:
                    break
                    
        if counter >= nr:  # check if value has occurred at least nr times
            naughty_list.append(current_value)
            a = str(start_date.strftime('%Y-%m-%d %H:%M:%S'))  # format start date as string
            first_date.append(a)
            last_date.append(last_seen_date.strftime('%Y-%m-%d %H:%M:%S'))  # format end date as string and add to list
            pairs = zip(naughty_list, first_date, last_date)
        else:
            pass 
    
    return naughty_list, first_date, last_date, list(pairs)



def flatline_lookup(data, tuplist, window=5):
    
    '''Function specifically for the tuple list output of flatlinef function.
        
        Prints dataframe at flatline values with extra values surrounding flatline given by window.  
    '''

    for _, second, third in tuplist:
        
        firstit = second
        lastit = third

        first_time = dt.datetime.strptime(firstit, "%Y-%m-%d %H:%M:%S")
        last_time = dt.datetime.strptime(lastit, "%Y-%m-%d %H:%M:%S")
        first = first_time - dt.timedelta(minutes=window)
        last  = last_time + dt.timedelta(minutes=window)
        display(data[(data['time']>=first)&(data['time']<=last)])
        






def qc_clean_data_loader(file_path="/Users/michelstahli/Boulder AIR/IDAT/LLG/2022/ch4", 
                         start_time=None, end_time=None, time_column='time', headerer=1):
    
    '''
    Loads data from entire folder, converts to UTC datetime, drops duplicates (first). 
        
    Assign to variable - Returns data ready to do stuff with! Doesn't touch NaNs. 
    Time is inclusive.    
        
    '''
    
    
    
    home = os.path.expanduser("~")


    all_files = glob.glob(file_path + "/*.csv")
    data = pd.concat((pd.read_csv(f, header=headerer) 
                                  for f in all_files), ignore_index=True) 
    data['unix_utc_time'] = data[time_column]
    data[time_column] = pd.to_datetime(data[time_column], unit='s', utc=True)
    data[time_column] = data[time_column].dt.tz_localize(None)
    #data['time'] = data['time'].dt.tz_localize('UTC')
    data = data.drop_duplicates()
    data = data.drop_duplicates(subset=time_column)
    if start_time and end_time:
        data = data[(data[time_column]>=start_time)&(data[time_column]<=end_time)]  
    elif start_time:
        data = data[(data[time_column]>=start_time)]
    elif end_time:
        data = data[(data[time_column]<=end_time)]
    else:
        data = data
    data = data.sort_values(time_column, ascending=True)
    return data
    
    
    
def qc_year_intro_analysis(data, location, year, time_column):
    
    '''
    Produces introductory analysis for Data QC for a single station.
    It prints number of missing values, max, min, LDL, 
    and informs user of LDL or general lower bound breach. 
    Graphs are produced (one or two) with LDL and/or half LDL depending on column/species. 
    Second graph is from 40th percentile of values down.
    
    Uses List and Dictionary from Common_Constants2
    It should produce accurate results even if data input is a dataset of merged folders, 
    HOWEVER, Missing Data will be incorrect depending on how you joined the data. 
    
    Arguments= 
    
    data: Dataframe 
    location: Station name, such as 'BNP', 'LUR' etc. 
    year: year of data. This is for graph purposes. ***********Consider making this optional***********
    
    Output is print statements and plots. Currently returns nothing. 
    '''

                
            
    columns = list(data.columns)
    columns = [c for c in columns if c not in UNDESIRED_LIST]
    
    low_dict = SITE_LOW_DICT[location]
    
    unit_dict = UNIT_DICT
    #ALSO REMOVE UNDESIRABLES************************************
    #print(low_dict)
    high_missing = []
    high_missing_name = []
    for column in columns:
        time.sleep(0.3)
        #print(colum)
        missing_fract = data[data[column].isna()==True].shape[0]/data.shape[0]
        if missing_fract>=0.15:
            high_missing.append((100*round((data[data[column].isna()==True].shape[0]/data.shape[0]), 5)))
            high_missing_name.append(column)    
            
        
        if column in low_dict:
            
                    
            if column not in ['co2_ppm', 'ch4', 'ch4_2', 'new_ch4', 'old_ch4']:
                ldl = low_dict[column]
                half_ldl = 0.5*ldl
                low_list = [half_ldl, ldl]
                low_col_data = data[data[column]<=ldl]
                naughty = set(low_col_data[column])-set(low_list) 

                print('\n'+'Column ',column)
                if column == 'co2_ppm':
                    print('Upper Bound is 700')
                    print('Max = '+str(data[column].max()))

                print('\n'+'Low/LDL is',str(ldl))
                print('Min is '+str(data[column].min()))

                if len(naughty):
                    print('Error!!! LDL process or Low Value Breach!!!')
                    print('Error!!! LDL process or Low Value Breach!!!')
                    print('Set= '+str(naughty))

                print('\n'+'Missing values: '+str(data[data[column].isna()==True].shape[0]))
                print('\n'+'Missing Percent: '+str(100*round((data[data[column].isna()==True].shape[0]/data.shape[0]), 5))+'%')
                
                    
                print('\n'+'1/2 LDL corrections: '+str(data[data[column]==half_ldl].shape[0]))
                print('\n'+'1/2 LDL corrections Percent: ' 
                      +str(100*round((data[data[column]==half_ldl].shape[0]/data.shape[0]), 4))+'%')
                print('\n'+'Le Graph'+'\n')

                sns.set_theme()
                plt.rcParams["figure.figsize"] = (16,11)
                sns.scatterplot(data=data, x=time_column, y=column, legend=False, edgecolor=None)
                plt.title(column, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', alpha=0.6)
                #plt.axhline(half_ldl, color='blue', linestyle='-')
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.2*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                #plt.text(fr'{year}-02-10 00:00:00', half_ldl+(0.2*data[column].std()), fr'Low/LDL = {half_ldl}', 'blue')
                plt.show()

                print('\n')
                
                sns.scatterplot(data=data[data[column]<data[column].quantile(0.99985)], x=time_column, y=column, legend=False, edgecolor=None)
                no_peak_title = column+' Below 99.985th Percentile'
                plt.title(no_peak_title, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', alpha=0.6)
                #plt.axhline(half_ldl, color='blue', linestyle='-')
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.2*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                #plt.text(fr'{year}-02-10 00:00:00', half_ldl+(0.2*data[column].std()), fr'Low/LDL = {half_ldl}', 'blue')
                plt.show()

                print('\n')
                
                sixth_percentile_above_ldl = data[column][data[column] > ldl].quantile(0.06)
                sns.scatterplot(data=data[data[column]<sixth_percentile_above_ldl], x=time_column, y=column, legend=False, edgecolor=None)
                second_title = column+' ldl+5th percentile and below'
                plt.title(second_title, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', label=str(ldl), alpha=0.6)
                plt.axhline(half_ldl, color='orange', linestyle='-')
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.05*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                
                plt.show()

                print('\n')
                try:
                    val, fd, ld, tup = flatlinef(df=data, column_name=column, nr=6)   
                    print(len(val))
                    print('First 6 values of flatlines >= 7 in length: '+str(val[0:5]))
                except:
                    print('Error, or No flatlines found that >= 7 in length.')
                print('              ------------------------------------------------------------------------------')
                print('\n')
                print('\n')
                print('\n')
                
            else:
                ldl = low_dict[column]    
                low_list = [ldl]
                low_col_data = data[data[column]<=ldl]
                naughty = set(low_col_data[column])-set(low_list) 

                print('\n'+'Column ',column)
                if column == 'co2_ppm':
                    print('Upper Bound is 700')
                    print('Max = '+str(data[column].max()))

                print('\n'+'Low/LDL is',str(ldl))
                print('Min is '+str(data[column].min()))

                if len(naughty):
                    print('Error!!! LDL process or Low Value Breach!!!')
                    print('Error!!! LDL process or Low Value Breach!!!')
                    print('Set= '+str(naughty))

                print('\n'+'Missing values: '+str(data[data[column].isna()].shape[0]))
                print('\n'+'Missing Percent: '+str(100*round((data[data[column].isna()==True].shape[0]/data.shape[0]), 5))+'%')
                #print('\n'+'1/2 LDL corrections: '+str(data[data[column]==half_ldl].shape[0]))
                #print('\n'+'1/2 LDL corrections Percent: ' 
                      #+str(100*round((data[data[column]==half_ldl].shape[0]/data.shape[0]), 4))+'%')
                print('\n'+'Le Graph'+'\n')

                sns.set_theme()
                plt.rcParams["figure.figsize"] = (16,11)
                sns.scatterplot(data=data, x=time_column, y=column, legend=False, edgecolor=None)
                plt.title(column, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', alpha=0.6) 
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.2*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                plt.show()
                
                
                print('\n')
                sns.scatterplot(data=data[data[column]<data[column].quantile(0.99985)], x=time_column, y=column, legend=False, edgecolor=None)
                no_peak_title = column+' Below 99.985th Percentile'
                plt.title(no_peak_title, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', alpha=0.6)
                #plt.axhline(half_ldl, color='blue', linestyle='-')
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.2*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                #plt.text(fr'{year}-02-10 00:00:00', half_ldl+(0.2*data[column].std()), fr'Low/LDL = {half_ldl}', 'blue')
                plt.show()

                print('\n')

                #data[data[column]<data[column].quantile(.28)].plot(x=time_column, y=column, legend=False)
                sixth_percentile_above_ldl = data[column][data[column] > ldl].quantile(0.06)
                sns.scatterplot(data=data[data[column]<sixth_percentile_above_ldl], x=time_column, y=column, legend=False, edgecolor=None)
                second_title = column+' ldl+5th percentile and below'
                plt.title(second_title, fontsize=23)
                plt.xticks(rotation=30, fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel(time_column, fontsize=17)
                if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
                plt.axhline(ldl, color='orange', linestyle='-.', alpha=0.6) 
                #plt.text(fr'{year}-02-10 00:00:00', ldl+(0.05*data[column].std()), fr'Low/LDL = {ldl}', color='blue')
                plt.show()

                print('\n')
                try:
                    val, fd, ld, tup = flatlinef(df=data, column_name=column, nr=6)   
                    print(len(val))
                    print('First 6 values of flatlines >= 7 in length: '+str(val[0:5]))
                except:
                    print('Error, or No flatlines found that >= 7 in length.')
                
                print('              ------------------------------------------------------------------------------')
                print('\n')
                print('\n')
                print('\n')
                
                        
        else:
            print('\n'+'Column ',column)
            print('Max = '+str(data[column].max()))
            print('Min is '+str(data[column].min()))

            print('\n'+'Missing values: '+str(data[data[column].isna()==True].shape[0]))
            print('\n'+'Missing Percent: '+str(100*round((data[data[column].isna()==True].shape[0]/data.shape[0]), 5))+'%')
            print('\n'+'Le Graph'+'\n')

            sns.set_theme()
            plt.rcParams["figure.figsize"] = (16,11)
            sns.scatterplot(data=data, x=time_column, y=column, legend=False, edgecolor=None)
            plt.title(column, fontsize=23)
            plt.xticks(rotation=30, fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(time_column, fontsize=17)
            if column in unit_dict:
                    units = unit_dict[column]
                    yl = column+' '+units
                    plt.ylabel(yl, fontsize=17)
            plt.show()

            print('\n')
            try:
                    val, fd, ld, tup = flatlinef(df=data, column_name=column, nr=6)   
                    print(len(val))
                    print('First 6 values of flatlines >= 7 in length: '+str(val[0:5]))
            except:
                    print('Error, or No flatlines found that >= 7 in length.')
            
            print('              ------------------------------------------------------------------------------')
            print('\n') 
            print('\n')
            print('\n')
    print('Missing data > 15% for folder')
    print(list(zip(high_missing, high_missing_name)))   
    print('\n')
    print('End of Folder')
    print('      ----------------------------------------'+
          '--------------------------------------------------------------------')    
    
    
def qc_intro_analysis(data, location, year, time_column, time_zone):
    
    if time_zone == 'utc':
        data=data
        qc_year_intro_analysis(data, location, year, time_column)
        
    elif time_zone == 'colorado':
        data = to_denver_tz_func(df = data,  time_column_header=time_column)
        qc_year_intro_analysis(data, location, year, time_column)
        
    else:
        print('No "time_zone" given. Remember the input must be in UTC datetime format already.')
    
    
    
def wind_nancombo_check(data, wdr_name, wsp_name):
    '''
    Checks the issue combinations of one column having a missing value while the other does not 
    for wind direction and windspeed. 
    
    data= dataframe
    wdr_name = wind direction column name
    wsp_name = wind speed column name 
    
    Runs as both a print statement and can return two dataframe subsets containing both types of missing combinations.
    
    Suggested Use:
    a, b = wind_nancombo_check(m, 'wdr_avg', 'wsp_avg_ms')
    wind_nancombo_check(m, 'wdr_avg', 'wsp_avg_ms')
    
    
    '''
    
    print('Wind Direction missing: '+str(data[data[wdr_name].isna()==True].shape[0]))
    print('Wind Speed missing: '+str(data[data[wsp_name].isna()==True].shape[0]))
    print('Wind Speed NaN, Wind Direction non-NaN '+
          str(data[(data[wdr_name].isna()==False)&(data[wsp_name].isna()==True)].shape[0]))
    print('Wind Direction NaN, Wind Speed non-NaN '+
          str(data[(data[wdr_name].isna()==True)&(data[wsp_name].isna()==False)].shape[0]))
    wspnan = data[(data[wdr_name].isna()==False)&(data[wsp_name].isna()==True)]
    wdrnan = data[(data[wdr_name].isna()==True)&(data[wsp_name].isna()==False)]
    print('wspnan')
    display(wspnan)
    print('\n')
    print('wdrnan')
    display(wdrnan)
    
    
    
    
def notebook_quick_graph(data, species, horizontal_value=None):  
    sns.set_theme()
    start = str(data['time'].min())
    end = str(data['time'].max())
    
    unit_dict = UNIT_DICT
    trange= start+"-"+end
    plt.close()
    if horizontal_value:
        
        
        title = species+' with line at '+str(horizontal_value)+'\n'+trange
        data.plot(x='time', y=species, legend=False)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=30)
        
        if species in unit_dict:
            units = unit_dict[species]
            yl = species+' '+units
            plt.ylabel(yl, fontsize=14)
        plt.axhline(horizontal_value, color='blue', linestyle='-.')
        plt.show()                        
                                
    else:
        title = species+'\n'+trange
        
        data.plot(x='time', y=species, legend=False)
        if species in unit_dict:
            units = unit_dict[species]
            yl = species+' '+units
            plt.ylabel(yl, fontsize=14)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=30)
        plt.show() 

        
        
        
def notebook_quick_scatter(data, species, below_value, horizontal_value=None):  
    sns.set_theme()
    start = str(data['time'].min())
    end = str(data['time'].max())
    
    unit_dict = UNIT_DICT
    trange= start+"-"+end
    plt.close()
    if horizontal_value:
        
        
        title = species+' with line at '+str(horizontal_value)+'\n'+trange
        data[data[species]<below_value].plot(x='time', y=species, legend=False, kind='scatter')
        plt.title(title, fontsize=14)
        plt.xticks(rotation=30)
        
        if species in unit_dict:
            units = unit_dict[species]
            yl = species+' '+units
            plt.ylabel(yl, fontsize=14)
        plt.axhline(horizontal_value, color='blue', linestyle='-.')
        plt.show()                        
                                
    else:
        title = species+' below '+str(below_value) +'\n'+trange
        fresh = data[data[species]<below_value]
        fresh.plot(x='time', y=species, legend=False, kind='scatter')
        if species in unit_dict:
            units = unit_dict[species]
            yl = species+' '+units
            plt.ylabel(yl, fontsize=14)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=30)
        plt.show()         
        
        




def outlier_detection_plot_tukey(data, column, threshold=2):
    """
    Automates peak/outlier detection in a pandas time series dataframe using the Tukey method and plots the data and outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    # Plot the data and outliers
    
    plt.plot(data['time'], data[column], label="Data")
    plt.scatter(outliers['time'], outliers[column], c="red", label="Outliers")
    plt.title(column)
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()
    plt.show()

    #return outliers


def outlier_detection_plot_zscore(data, column, threshold=3.5):
    """
    Automates peak/outlier detection in a pandas time series dataframe.
    """
    z_score = np.abs(data[column] - data[column].mean()) / data[column].std()
    outliers = data[z_score > threshold]
    
    
    sns.set_theme()
    plt.rcParams["figure.figsize"] = (14,9)
    plt.plot(data['time'], data[column], label="Data")
    plt.scatter(outliers['time'], outliers[column], c="red", label="Outliers")
    plt.title(column)
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()
    plt.show()
    
    #return outliers
    
    
    
    

def voc_rearrange_columns(df):
    columns = list(df.columns)
    a, b, c, d, e, f, g, h, i, j, k, l = 'unix_utc_time', 'time', 'ethane',  'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene'
    if a in columns and b in columns and c in columns and d in columns:
        # Move the four columns to the front
        columns.remove(a)
        columns.remove(b)
        columns.remove(c)
        columns.remove(d)
        columns.remove(e)
        columns.remove(f)
        columns.remove(g)
        columns.remove(h)
        columns.remove(i)
        columns.remove(j)
        columns.remove(k)
        columns.remove(l)
        new_columns = [a, b, c, d, e, f, g, h, i, j, k, l] + sorted(columns)
        # Return the dataframe with the rearranged columns
        return df[new_columns]
    else:
        raise ValueError("Input dataframe must have columns unix_utc_time, time, and the 6 voc bosses.")
   





     


def voc_missing_csv_export(df, file_path, site_name, time_period_name):
    
    data = voc_rearrange_columns(df)
    
    if site_name=='CCM':
    
    
        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = num_missing_values == 1
        data_with_one_missing_values = data[mask]
        data_with_one_missing_values = voc_rearrange_columns(data_with_one_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = (num_missing_values >= 2) & (num_missing_values <= 3)
        data_with_twothree_missing_values = data[mask]
        data_with_twothree_missing_values = voc_rearrange_columns(data_with_twothree_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = num_missing_values >= 4
        data_with_4plus_missing_values = data[mask]
        data_with_4plus_missing_values = voc_rearrange_columns(data_with_4plus_missing_values)
  
    else:
        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = num_missing_values == 1
        data_with_one_missing_values = data[mask]
        data_with_one_missing_values = voc_rearrange_columns(data_with_one_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = (num_missing_values >= 2) & (num_missing_values <= 3)
        data_with_twothree_missing_values = data[mask]
        data_with_twothree_missing_values = voc_rearrange_columns(data_with_twothree_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = num_missing_values >= 4
        data_with_4plus_missing_values = data[mask]
        data_with_4plus_missing_values = voc_rearrange_columns(data_with_4plus_missing_values)
    
    
    #Export
    one_filepath = file_path + site_name + ' ' + time_period_name + ' one voc boss missing'+'.csv'
    data_with_one_missing_values.to_csv(one_filepath, index=False)

    two_three_filepath = file_path + site_name + ' ' + time_period_name + ' two_three voc boss missing'+'.csv'
    data_with_twothree_missing_values.to_csv(two_three_filepath, index=False)
    
    fourplus_filepath = file_path + site_name + ' ' + time_period_name + ' fourplus voc boss missing'+'.csv'
    data_with_4plus_missing_values.to_csv(fourplus_filepath, index=False)
    
    

    
    
    
def voc_missing_view(df, ccm = False):
    
    data = voc_rearrange_columns(df)
    
    if ccm == True:
        print('This is for CCM/including Toluene, Benzene, o and mp-xylenes, as the VOC Bosses')
    
    
        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = num_missing_values == 1
        data_with_one_missing_values = data[mask]
        data_with_one_missing_values = voc_rearrange_columns(data_with_one_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = (num_missing_values >= 2) & (num_missing_values <= 3)
        data_with_twothree_missing_values = data[mask]
        data_with_twothree_missing_values = voc_rearrange_columns(data_with_twothree_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 
                                   'i-pentane', 'n-pentane', 'toluene', 'benzene', 'o-xylene', 'm&p-xylene']].isna().sum(axis=1)
        mask = num_missing_values >= 4
        data_with_4plus_missing_values = data[mask]
        data_with_4plus_missing_values = voc_rearrange_columns(data_with_4plus_missing_values)
  
    else:
        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = num_missing_values == 1
        data_with_one_missing_values = data[mask]
        data_with_one_missing_values = voc_rearrange_columns(data_with_one_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = (num_missing_values >= 2) & (num_missing_values <= 3)
        data_with_twothree_missing_values = data[mask]
        data_with_twothree_missing_values = voc_rearrange_columns(data_with_twothree_missing_values)

        num_missing_values = data[['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane']].isna().sum(axis=1)
        mask = num_missing_values >= 4
        data_with_4plus_missing_values = data[mask]
        data_with_4plus_missing_values = voc_rearrange_columns(data_with_4plus_missing_values)
    
    
    print('Rows with only one missing value among VOC bosses:')
    display(data_with_one_missing_values)
    print('\n')
    print('Rows with only 2-3 missing values among VOC bosses:')
    display(data_with_twothree_missing_values)
    print('\n')
    print('Rows with 4 or more missing values among VOC bosses:')
    display(data_with_4plus_missing_values)
    print('\n')
    
    
    
def folders_missing_dates(folder_list):
    
    for i, folder in enumerate(folder_list):
        
        print(str(i))
            
        
        date_range = pd.date_range(start=folder['time'].min(), end=folder['time'].max(), freq='S')

        # Reindex the dataframe with the complete range of dates
        folder = folder.set_index('time').reindex(date_range).reset_index()
        
        # Create a Boolean mask indicating which values in the time column are missing
        mask = folder['index'].isna()

        # Convert the Boolean mask to integers
        int_mask = mask.astype(int)

        # Calculate the difference between consecutive elements in the Boolean mask
        diff_mask = int_mask.diff()

        # Extract the start and end indices of missing chunks
        start_indices = diff_mask[diff_mask == 1].index
        end_indices = diff_mask[diff_mask == -1].index

        
        
        start_dates = folder.loc[start_indices, 'index']
        end_dates = folder.loc[end_indices, 'index']

        # Combine the start and end dates into a list of tuples
        date_tuples = list(zip(start_dates, end_dates))
        # Calculate the size of each missing chunk
        sizes = end_indices - start_indices + 1

        # Summarize the information about the missing chunks
        missing_chunks = sizes.value_counts()
        
        print('How many and size')
        print(missing_chunks)
        print('\n'+'First and Last')
        print(date_tuples)
        print('\n')
   
        

    
def sigfig_column_renamer(df):

        df = df.rename(columns=RENAME_MAPPING, inplace=True)
        
        return df

    
def sigfig_pre_export(df):
    
    #NOTE THIS REQUIRES RENAMED DF. USE COLUMN_RENAMER FIRST 
    
    precision_mapping = DECIMAL_MAPPING
    
   
    
    for col in [x for x in list(df.columns) if x not in UNDESIRED_LIST]:
        if col in precision_mapping:
            precision = int(precision_mapping[col])
            
            df[col] = df[col].astype(float)
            df[col] = df[col].round(precision).apply(lambda x: f"{x:.{precision}f}")
        if col in VOC_LIST:
            df[col] = df[col].round(3).apply(lambda x: "{:.3f}".format(x))
            


def big_export_renamer(df):
    
    print(df.columns)
    df = df.rename(columns=POST_SIGFIG_RENAME_MAPPING)
    print(df.columns)
    
    return df
    
    
    
def big_export_reorder(df, site):
    head1 = HEAD1_BIG_DICT[site]
    df = df[head1]
    
    return df
    
    
    
def big_export_headers(df, site):
    head2 = HEAD2_BIG_DICT[site]
    
    header=[df.columns, head2]
 
    df.columns=header

    return df


def radon_export_reorder(df):
    df = df[CCF_RADON_CSV_HEADER1]
    
    return df


def radon_export_headers(df):
    
    header=[df.columns, CCF_RADON_CSV_HEADER2]
 
    df.columns=header
    
    return df



def voc_export_reorder(df, site):
    head1 = HEAD1_VOC_DICT[site]
    print(head1)
    print(df.columns)
    df = df[head1]
    
    return df


def voc_export_headers(df, site):
    head2 = HEAD2_VOC_DICT[site]
    header=[df.columns, head2]
 
    df.columns=header
    
    return df















    
    
    
    