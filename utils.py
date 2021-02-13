# -*- coding: utf-8 -*-
"""
This is utils.py to define basic functions to be used by EDA and Modelling work's notebook
author: Harsh Shah
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# plt.style.use('seaborn-dark')
# color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]



def reduce_mem_usage(df, verbose=True):
    """
    Return pandas dataframe with reduced memory usage 
    :param
        df:  pandas dataframe
    :return:
        df: pandas dataframe with reduced memory usage 
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#Help taken from Kernel: https://www.kaggle.com/ysjf13/cis-fraud-detection-visualize-feature-engineering/notebook


def numeric_features(df):
    """
    Return numeric features of dataframe 
    :param
        df:  pandas dataframe
    :return:
        summary: list that contains numerical features
    """
    numeric_col = df.select_dtypes(include=np.number).columns.tolist()
    return numeric_col


def categorical_features(df):
    """
    Return categorical features of dataframe 
    :param
        df:  pandas dataframe
    :return:
        summary: list that contains categorical features
    """
    categorical_col = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_col

def data_summary(df):
    """
    Return summary sof the dataframe e.g. Missing values, unique values, first and last values of the dataset
    :param
        df:  pandas dataframe
    :return:
        summary: dataframe that contains summary of  df
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing Data'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Last Value'] = df.iloc[-1].values

    return summary

def count_fraud_plot(df, col, bins, patch_fontsize=8, axes_rotation_value=45):
    """
    Return count distribution plot against categorical column and the fraud percent of each category
    :param
        df:  pandas dataframe
        col: categorical feature
        bins: number of bins on plot
        patch_fontsize: font size for patch on top of each bar
        axes_rotation_value: x and y axis rotation for better readability
    :return
    """

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    sns.countplot(x=col, data=df,
                  order=df[col].value_counts().iloc[:bins].index, ax=ax[0])

    # ax[0].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[0].set_title(
        f"Frequency of {col} values", fontsize=16)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=axes_rotation_value)

    for p in ax[0].patches:
        height = p.get_height()
        ax[0].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height / df.shape[0] * 100:.2f}%',
                   ha='center', fontsize=patch_fontsize,rotation=axes_rotation_value)

    sns.barplot(x=col, y='Fraud', data=tmp,
                order=df[col].value_counts().iloc[:bins].index, ax=ax[1])

    ax[1].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[1].set_title(
        f"Fraud Percentage of {col} values", fontsize=16)
    ax[1].set_ylabel("Percent", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=axes_rotation_value)
    plt.subplots_adjust(hspace=.4, top=0.9)

    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height:.2f}%',
                   ha='center', fontsize=patch_fontsize,rotation=axes_rotation_value)
    plt.subplots_adjust(hspace=.4, top=1.1)
    plt.show();
    
def handle_redundant_email_domains(df,email_domains_list):
    """
    Handles the different names of email domains and maps it to single name as category
    :param
        df:  pandas dataframe
        email_domains_list: list of columns of email_domains
    :return
    """
    for email in email_domains_list:
        df.loc[df[email].isin(['gmail.com', 'gmail']), email] = 'Google Mail'
        
        df.loc[df[email].isin(['yahoo.com', 'ymail.com', 'yahoo.com.mx',
                             'yahoo.co.jp', 'yahoo.fr', 'yahoo.co.uk',
                             'yahoo.es', 'yahoo.de']), email] = 'Yahoo Mail'
        
        df.loc[df[email].isin(['hotmail.com', 'outlook.com', 'msn.com',
                             'live.com', 'live.com.mx', 'outlook.es',
                             'hotmail.fr', 'hotmail.co.uk', 'live.fr',
                             'hotmail.es', 'hotmail.de']), email] = 'Microsoft mail'
        
        df.loc[df[email].isin(['icloud.com', 'me.com', 'mac.com']), email] = 'Apple mail'
        
        df.loc[df[email].isin(df[email].value_counts()[df[email].value_counts() <= 1000].index), email] = 'Others'
        df['P_emaildomain'].fillna("Unknown", inplace=True)
        df['R_emaildomain'].fillna("Unknown", inplace=True)


def handle_device_info(df):
    """
    Handles the different names of devices companies and maps it to single name as category
    :param
        df:  pandas dataframe
    :return
    """
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['DeviceInfo'].str.contains('SM', na=False), 'DeviceInfo'] = 'Samsung'
    df.loc[df['DeviceInfo'].str.contains('SAMSUNG', na=False), 'DeviceInfo'] = 'Samsung'
    df.loc[df['DeviceInfo'].str.contains('GT-', na=False), 'DeviceInfo'] = 'Samsung'
    df.loc[df['DeviceInfo'].str.contains('Moto G', na=False), 'DeviceInfo'] = 'Motorola'
    df.loc[df['DeviceInfo'].str.contains('Moto', na=False), 'DeviceInfo'] = 'Motorola'
    df.loc[df['DeviceInfo'].str.contains('moto', na=False), 'DeviceInfo'] = 'Motorola'
    df.loc[df['DeviceInfo'].str.contains('LG-', na=False), 'DeviceInfo'] = 'LG'
    df.loc[df['DeviceInfo'].str.contains('rv:', na=False), 'DeviceInfo'] = 'RV'
    df.loc[df['DeviceInfo'].str.contains('HUAWEI', na=False), 'DeviceInfo'] = 'Huawei'
    df.loc[df['DeviceInfo'].str.contains('ALE-', na=False), 'DeviceInfo'] = 'Huawei'
    df.loc[df['DeviceInfo'].str.contains('-L', na=False), 'DeviceInfo'] = 'Huawei'
    df.loc[df['DeviceInfo'].str.contains('Blade', na=False), 'DeviceInfo'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.contains('BLADE', na=False), 'DeviceInfo'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.contains('Linux', na=False), 'DeviceInfo'] = 'Linux'
    df.loc[df['DeviceInfo'].str.contains('XT', na=False), 'DeviceInfo'] = 'Sony'
    df.loc[df['DeviceInfo'].str.contains('HTC', na=False), 'DeviceInfo'] = 'HTC'
    df.loc[df['DeviceInfo'].str.contains('ASUS', na=False), 'DeviceInfo'] = 'Asus'

    df.loc[df['DeviceInfo'].isin(df['DeviceInfo'].value_counts()[df['DeviceInfo'].value_counts() < 1000].index), 'DeviceInfo'] = "Others"
        
def missing_data(df):
    """
    Returns missing data information as dataframe and PLOTS column vs missing ratio barplot
    :param
        df:  pandas dataframe
    :return
        missing_data: pandas dataframe with columns 'total' and 'percent'
    """
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    plt.subplots(figsize=(20,8))
    ax = sns.barplot(x = missing_data.index, y = missing_data['Percent'])#, orient='h')
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Missing %', fontsize=20)
    plt.xticks(rotation=60)
    plt.show()

    return missing_data

def get_missing_columns(missing_data):
    """
    Returns columns names as list that containes missing data 
    :param
        missing_data : return of missing_data(df)
    :return
        list: list containing columns with missing data
    """
    missing_data = missing_data[missing_data['percent'] > 0]
    missing_columns = missing_data.index.tolist()
    return missing_columns

def data_cleaning(df):
    """
    Return a cleaned processed dataframe after dropping dirty data
    :param
        df : pandas df (merged train_test preferably)
    :return
        cleaned_df: cleaned pandas dataframe
    """
    # Dropping columns with > 90% missing data
    null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    df = df.drop(null_cols,axis=1)
    print("List of dropped columns that had > 90% missing data")
    print(null_cols)
    print("="*100)
    
    # Reducing memory usage
    df = reduce_mem_usage(df)
    
    # Dropping Highly Correlated Values here
    ## Sorting the numerical columns by name
    numeric_cols = numeric_features(df)
    numeric_cols.sort()
    
    ## Creating correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    ## Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    ## Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    print('Showing the first 20 elements of to_drop (out of {}):\n{}'.format(len(to_drop),to_drop[:20]))
    print("="*100)
    
    ## Now dropping them
    to_drop_notV = to_drop[:14]
    to_drop_V = to_drop[14:]
    for col in to_drop_notV:
        na_proportion = df[col].isna().sum()/df.shape[0]
        if na_proportion >= 0.4:
            df = df.drop(col,axis=1)
            print('Column {} was dropped'.format(col))
    df = df.drop(columns = to_drop_V)
    print('\n{} V columns were dropped'.format(len(to_drop_V)))
    
    ## Excluding the removed features from the definition of num_cols
    cat_cols = categorical_features(df)
    numeric_cols = list(set(numeric_cols) - set(['D2','D6']) - set(to_drop_V))
    
    ## Replacing NaN values for model training
    df[cat_cols] = df[cat_cols].astype('str').replace('nan','unknown').astype('category')
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    
    
    return df