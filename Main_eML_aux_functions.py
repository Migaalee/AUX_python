# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:52:41 2022

@author: mmiskinyte
"""

### AUX FUNCTIONS FOR eML project for low frequency mutation detection



import sys
import os
import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
import itertools
from bokeh.layouts import gridplot
import aplanat
from aplanat import bars
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack

import re


### AUX functions

def expand_rows(df, column_to_expand, other_columns):
    # Initialize an empty list to hold the dictionaries
    expanded_rows = []
    
    for _, row in df.iterrows():
        # Handle NaN values in the column to expand
        if pd.isna(row[column_to_expand]):
            values_to_expand = ['']  # or some other placeholder you prefer
        else:
            values_to_expand = str(row[column_to_expand]).split(',')
        
        # Replicate rows for each value in the column to expand
        for value in values_to_expand:
            new_row = row.to_dict()
            new_row[column_to_expand] = value
            
            # Handle other columns that have comma-separated values or NaN values
            for col in other_columns:
                if col in row:  # Check if the column exists in the row
                    if pd.isna(row[col]):
                        new_row[col] = ''  # or some other placeholder you prefer
                    elif ',' in str(row[col]):
                        split_values = str(row[col]).split(',')
                        new_row[col] = split_values[values_to_expand.index(value)] if len(split_values) > values_to_expand.index(value) else ''
                    else:
                        new_row[col] = row[col]  # Use the same value if it's not a list
                else:
                    new_row[col] = ''  # or some other placeholder if column doesn't exist
            
            # Add the new row dictionary to the list
            expanded_rows.append(new_row)
    
    # Convert the list of dictionaries to a DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


def parse_vcf_freebayes(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe with expanded ALT alleles.
    Each ALT allele is given its own row with associated information from other columns.
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None)).astype(dtype)
            except:
                pass
    vcf.drop(columns=['INFO'], inplace=True)
    
    if info_cols:
        comma_separated_columns = [
        'ALT', 'QUAL', 'FILTER', 'FORMAT', 'GT', 'AB', 'ABP', 'AF', 'AO',
        'DPRA', 'LEN', 'MQM', 'PAIRED', 'PAO', 'PQA', 'QA', 'RPL', 'RPP', 'RPR',
        'RUN', 'SAF', 'SAP', 'SAR', 'TYPE', 'AC', 'CIGAR', 'EPP', 'MEANALT'
    ] + list(info_cols.keys())
    else:
        comma_separated_columns = [
        'ALT', 'QUAL', 'FILTER', 'FORMAT', 'GT', 'AB', 'ABP', 'AF', 'AO',
        'DPRA', 'LEN', 'MQM', 'PAIRED', 'PAO', 'PQA', 'QA', 'RPL', 'RPP', 'RPR',
        'RUN', 'SAF', 'SAP', 'SAR', 'TYPE', 'AC', 'CIGAR', 'EPP', 'MEANALT'
    ]
 
    # Expand the rows for multiple ALT alleles
    expanded_vcf = expand_rows(vcf, 'ALT', comma_separated_columns)
    
    return expanded_vcf



def parse_vcf_bcftools(fname, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary, and each key-value pair is expanded into distinct columns.
    nrows: how many rows to read from the start of the header.
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)

    # Function to parse the INFO field into a dictionary
    def parse_info(info_str):
        info_dict = {}
        for item in info_str.split(";"):
            parts = item.split("=")
            if len(parts) == 2:
                key, value = parts
                info_dict[key] = value
        return info_dict

    # Apply parse_info function to each row in the INFO column
    vcf['INFO'] = vcf['INFO'].apply(parse_info)

    # Create a list of multi-valued key columns to drop later
    multi_key_columns = []

    # Parse columns with single keys
    single_keys = set()
    for info_dict in vcf['INFO']:
        single_keys.update(info_dict.keys())

    for key in single_keys:
        vcf[key] = vcf['INFO'].apply(lambda x: x.get(key, None))
        vcf[key] = pd.to_numeric(vcf[key], errors='ignore')

    # Create separate columns for multi-valued keys (e.g., PV4)
    for key in vcf['INFO'].apply(lambda x: list(x.keys())).explode().unique():
        if vcf[key].apply(lambda x: isinstance(x, str) and ',' in x).any():
            values = vcf[key].str.split(',', expand=True)
            num_columns = values.shape[1]
            for i in range(1, num_columns + 1):
                new_key = f'{key}_{i}'
                vcf[new_key] = values[i - 1]
                vcf[new_key] = pd.to_numeric(vcf[new_key], errors='ignore')
            multi_key_columns.append(key)  # Add the multi-valued key to the list

    # Drop the original multi-valued key columns
    vcf.drop(columns=multi_key_columns, inplace=True)

    # Drop the original INFO column if no longer needed
    vcf.drop(columns=['INFO','GT','FILTER','FORMAT','FQ' ], inplace=True)

    return vcf



def parse_vcf_varscan(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # Create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    
    # Parse FORMAT into column names
    format_cols = vcf['FORMAT'].str.split(":").iloc[0]
    
    # Split GT columns by ":" and create new columns with names from FORMAT
    gt_data = vcf['GT'].str.split(":", expand=True)
    gt_data.columns = format_cols
    
    vcf = pd.concat([vcf, gt_data], axis=1)
    vcf.drop(columns=['INFO','GT','FORMAT','FREQ' ], inplace=True)
    
    return vcf



def parse_vcf_strelka(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # Create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    
    # Parse FORMAT into column names
    format_cols = vcf['FORMAT'].str.split(":").iloc[0]
    
    # Split GT columns by ":" and create new columns with names from FORMAT
    gt_data = vcf['GT'].str.split(":", expand=True)
    gt_data.columns = format_cols
    
    vcf = pd.concat([vcf, gt_data], axis=1)
    vcf.drop(columns=['INFO','GT','FORMAT' ], inplace=True)
    
    return vcf


def parse_vcf_mutect(fname, info_cols=None, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP': int, 'CIGAR': str}, nrows=1000)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)

    def parse_info(info_str):
        info_dict = {}
        for item in info_str.split(";"):
            parts = item.split("=")
            if len(parts) == 2:
                key, value = parts
            else:
                key = parts[0]
                value = ""
            info_dict[key] = value
        return info_dict

    vcf['INFO'] = vcf['INFO'].apply(parse_info)

    if info_cols is not None:
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    return vcf



def confusion_matrix_m(dataset, gt_dataset, genome_size):
    # Convert POS columns to numpy arrays for faster processing
    d_POS = dataset["POS"].values
    g_POS = gt_dataset["POS"].values

    # Initialize confusion matrix
    matrix = np.zeros((2, 2))

    # Iterate through the dataset to calculate TP and FP
    for pos in d_POS:
        if pos in g_POS:
            matrix[0, 0] += 1  # True Positives
        else:
            matrix[0, 1] += 1  # False Positives

    # Calculate False Negatives
    for pos in g_POS:
        if pos not in d_POS:
            matrix[1, 0] += 1  # False Negatives

    # Calculate True Negatives
    # Assuming true negatives are all positions not called in the dataset and not present in the ground truth
    matrix[1, 1] = genome_size - len(np.unique(np.concatenate((d_POS, g_POS))))

    return matrix


def confusion_matrix_m1(dataset, gt_dataset, genome_size):
    # Convert POS and ALT columns to numpy arrays for faster processing
    d_POS = dataset["POS"].values
    d_ALT = dataset["ALT"].values
    g_POS = gt_dataset["POS"].values
    g_ALT = gt_dataset["ALT"].values

    # Initialize confusion matrix
    matrix = np.zeros((2, 2))
    
    # Print total number of variants called with this tool:
    
    total=len(np.unique(d_POS))

    # Iterate through the dataset to calculate TP and FP
    for i, alt in zip(d_POS, d_ALT):
        if i in g_POS:
            # Find the index of the matching POS in ground truth dataset
            index = np.where(g_POS == i)[0]
            print(index)
            if alt == g_ALT[index[0]]:
                matrix[0, 0] += 1  # True Positives
            else:
                matrix[0, 1] += 1  # False Positives
        else:
            matrix[0, 1] += 1  # False Positives

    # Calculate False Negatives
    for i, alt in zip(g_POS, g_ALT):
        if not (i in d_POS and alt in d_ALT):
            matrix[1, 0] += 1  # False Negatives

    # Calculate True Negatives
    # Assuming true negatives are all positions not called in the dataset and not present in the ground truth
    matrix[1, 1] = genome_size - len(np.unique(np.concatenate((d_POS, g_POS))))
    
    print("Total number of variants called are: ", str(total))

    return  matrix




def plot_custom_confusion_matrix(confusion_matrix, title="Conf Matrix"):
    """
    This function plots a custom 2x2 confusion matrix with specified colors by name and labels for TP, TN, FP, FN.

    Args:
    confusion_matrix (list of lists): A 2x2 confusion matrix in the format [[TP, FP], [FN, TN]]
    title (str): Custom title for the confusion matrix (optional)

    Returns:
    None
    """
    class_names = ['Positive', 'Negative']

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='gray', vmin=0, vmax=1, aspect='auto')  # Light gray background
    plt.title(title)  # Set the custom title

    tick_marks = [0, 1]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    labels = ["TP", "FP", "FN", "TN"]
    for i in range(2):
        for j in range(2):
            if i == j:
                color = 'green'  # TP and TN in limegreen
            else:
                color = 'red'    # FP and FN in pink
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment="center", color=color)
            plt.text(j, i-0.2, labels[i*2+j], horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Example usage with a custom title:
# Replace the example matrix with your actual 2x2 confusion matrix
# custom_matrix = [[42, 8], [12, 38]]
# plot_custom_confusion_matrix(custom_matrix, title="Confusion Matrix - My Title")




def metrics_m(matrix, calculate=None):
    """
    Function to calculate metrics for different variant callers based on confusion matrix.
    If no second argument is given, returns all the metrics. If a particular metric is necessary,
    define the argument in the following way.
    Example:
    precision = metrics_m(matrix, "precision")
    """
    TP = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Check if both precision and recall are zero for F1 calculation
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (recall * precision) / (recall + precision)

    accuracy = (TP + matrix[1, 1]) / np.sum(matrix) if np.sum(matrix) > 0 else 0

    metrics_values = [precision, recall, accuracy, f1]

    if calculate is None:
        return metrics_values
    elif calculate == "precision":
        return precision
    elif calculate == "recall":
        return recall
    elif calculate == "accuracy":
        return accuracy
    elif calculate == "f1":
        return f1


def TPR_FPR(dataset, gt_dataset, genome_size): #pass dataframe from variant caller and GT dataframe
    d_P=dataset["POS"].values
    g_P=gt_dataset["POS"].values
    matrix=np.zeros((2,2)) # matrix of 2 by 2
    matrix[1,1]=genome_size - len(gt_dataset) # True Negatives
    for i in d_P:
        if i in g_P:
            matrix[0,0]+=1 #True Positives
        elif i not in g_P:
            matrix[0,1]+=1 #False Positives
    for k in g_P:
      if k not in d_P:
        matrix[1,0]=+1 #False Negatives
    TPR=matrix[0,0]/(matrix[0,0]+matrix[1,0])
    FPR=matrix[0,1]/(matrix[1,1]+matrix[0,1])
    return TPR, FPR





# columns_to_remove = ['REF', 'REF_y']

# # # Remove the specified columns
# df_3 = df_3.drop(columns=columns_to_remove)

# for column in df_filtered.columns:
#         if 'REF' in column:
#             non_empty_count = df_filtered[column].dropna().shape[0]
#             print(f"Column '{column}' has {non_empty_count} non-empty rows.")




def data_imputation(df, threshold_missing_data,imputation_methods=None):
    
    
    '''
    This function is required for data clean up and imputation.
    First, it deletes columns that has a percentage of missing data that is above user defined threshold.
    Second, it 
    
    
    Example of how to use function:
        imputation_methods = {'column1': 'mean', 'column2': 42, 'column3': 'median'}
        df_new=remove_bad_columns(df, imputation_methods, threshold_missing_data=99.99)
    
    
    '''
    threshold=threshold_missing_data
    max_count = 0
    max_column = None
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_test = df.copy()
    
    # Remove all columns that has more than 99.99% values missing in rows
    
    percent_missing = df_test.isnull().mean() * 100

    columns_to_keep = percent_missing[percent_missing < threshold].index
    df_test = df_test[columns_to_keep]
    
    
    # Check which REF column (several were generated due to join by POS from different variant callers) has more rows and keep only that column
    ref_columns = [col for col in df_test.columns if 'REF' in col] 
    if len(ref_columns) > 1:
        for column in ref_columns: 
            non_empty_count = df_test[column].dropna().shape[0] 
            if non_empty_count > max_count:
                max_count = non_empty_count
                max_column = column

        if max_column:
            print(f"Column '{max_column}' with highest number of non-empty rows: {max_count} is kept")
            columns_to_drop = [col for col in ref_columns if col != max_column]
            df_test.drop(columns=columns_to_drop, inplace=True)
        
    elif len(ref_columns) == 1:
        print(f"Only one REF column '{ref_columns[0]}' found in the DataFrame. Keeping it.")
    else: 
        print("No 'REF' columns found.")
    
        
        
    # Now replace empty rows of chosen columns either with max value, mean value or a specific number
    if not imputation_methods:
        print("You did not impute any missing values.")
        return df_test

    for column, impute_info in imputation_methods.items():
        if column in df_test.columns:
            impute_method = impute_info['method']
            impute_type = impute_info['type']

            if impute_type == 'string':
                # String imputation
                df_test[column].fillna(impute_method, inplace=True)
            elif impute_type in ['float', 'numeric']:
                # Convert column to numeric, if possible
                df_test[column] = pd.to_numeric(df_test[column], errors='coerce')
                
                # Numeric imputation methods
                if impute_method == 'max':
                    df_test[column].fillna(df_test[column].max(), inplace=True)
                elif impute_method == 'min':
                    df_test[column].fillna(df_test[column].min(), inplace=True)
                elif impute_method == 'mean':
                    df_test[column].fillna(df_test[column].mean(), inplace=True)
                elif impute_method == 'median':
                    df_test[column].fillna(df_test[column].median(), inplace=True)
                else:  # Assuming impute_method is a specific number
                    df_test[column].fillna(float(impute_method), inplace=True)
            else:
                print(f"Imputation type '{impute_type}' not recognized for column '{column}'")
        else:
            print(f"Column '{column}' not found in the DataFrame.")
    
        
        # Check for remaining missing data
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0] 
    if missing_columns.empty: 
        print("All data successfully imputed.")
    else: 
        print("Columns with remaining missing data:") 
        for column, missing_percent in missing_columns.items(): 
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")        
         
    return df_test



def data_imputation2(df, threshold_missing_data, imputation_methods=None):
    '''
    This function performs data clean up and imputation.
    - Deletes columns with a percentage of missing data above a user-defined threshold.
    - Imputes missing data based on specified methods, defaulting to median ±1 standard deviation for columns with <5% missing data.

    Example of how to use function:
        imputation_methods = {'column1': {'method': 'mean', 'type': 'numeric'}, 
                              'column2': {'method': 42, 'type': 'numeric'}, 
                              'column3': {'method': 'median', 'type': 'numeric'}}
        df_new = data_imputation(df, imputation_methods, threshold_missing_data=99.99)
    '''
    threshold = threshold_missing_data

    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_test = df.copy()

    # Remove columns with missing values above threshold
    percent_missing = df_test.isnull().mean() * 100
    columns_to_keep = percent_missing[percent_missing < threshold].index
    df_test = df_test[columns_to_keep]

    # Impute missing values based on specified methods or default method
    for column in df_test.columns:
        if df_test[column].isnull().mean() * 100 < 5:
            if (imputation_methods is None) or (column not in imputation_methods):
                # Median imputation with random deviation
                median = df_test[column].median()
                std = df_test[column].std()
                random_deviation = np.random.uniform(-std, std, size=df_test[column].isnull().sum())
                df_test.loc[df_test[column].isnull(), column] = median + random_deviation
                continue

        if imputation_methods and column in imputation_methods:
            impute_info = imputation_methods[column]
            impute_method = impute_info['method']
            impute_type = impute_info['type']

            if impute_type == 'string':
                df_test[column].fillna(impute_method, inplace=True)
            elif impute_type in ['float', 'numeric']:
                df_test[column] = pd.to_numeric(df_test[column], errors='coerce')
                if impute_method == 'max':
                    df_test[column].fillna(df_test[column].max(), inplace=True)
                elif impute_method == 'min':
                    df_test[column].fillna(df_test[column].min(), inplace=True)
                elif impute_method == 'mean':
                    df_test[column].fillna(df_test[column].mean(), inplace=True)
                elif impute_method == 'median':
                    df_test[column].fillna(df_test[column].median(), inplace=True)
                else:  # Assuming impute_method is a specific number
                    df_test[column].fillna(float(impute_method), inplace=True)

    # Check for remaining missing data and report
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0]
    if missing_columns.empty:
        print("All data successfully imputed.")
    else:
        print("Columns with remaining missing data:")
        for column, missing_percent in missing_columns.items():
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")

    return df_test


def select_column_with_max_rows(df, column_keyword):
    max_count = 0
    max_column = None
    columns = [col for col in df.columns if column_keyword in col]

    if len(columns) > 1:
        for column in columns:
            non_empty_count = df[column].dropna().shape[0]
            if non_empty_count > max_count:
                max_count = non_empty_count
                max_column = column

        if max_column:
            print(f"Column '{max_column}' with highest number of non-empty rows: {max_count} is kept")
            columns_to_drop = [col for col in columns if col != max_column]
            df.drop(columns=columns_to_drop, inplace=True)

    elif len(columns) == 1:
        print(f"Only one '{column_keyword}' column '{columns[0]}' found in the DataFrame. Keeping it.")
    else:
        print(f"No '{column_keyword}' columns found.")

    return df




def data_imputation3(df, threshold_missing_data, imputation_methods=None, columns_to_remove=None):
    '''
    Performs data clean up and imputation on a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold_missing_data (float): Threshold for missing data in percent. Columns with a higher percentage of missing data will be removed.
    - imputation_methods (dict, optional): A dictionary specifying imputation methods for specific columns. The dictionary format should be:
        {'column_name': {'method': 'mean/median/max/min/specific_value', 'type': 'numeric/string'}}
      If no method is specified for a column with less than 5% missing data, median imputation with random deviation within one standard deviation is used.

    Returns:
    - DataFrame: The DataFrame after imputation.

    The function first removes columns with missing data above the specified threshold. It then applies imputation methods based on the 'imputation_methods' dictionary. For columns with numeric data and less than 5% missing values, where no specific method is provided, the function defaults to median imputation with a random deviation within one standard deviation.

    Example of how to use function:
        imputation_methods = {'column1': {'method': 'mean', 'type': 'numeric'}, 
                              'column2': {'method': 42, 'type': 'numeric'}, 
                              'column3': {'method': 'median', 'type': 'numeric'}}
        df_new = data_imputation(df, threshold_missing_data=99, imputation_methods=imputation_methods)
    '''
    

    # Create a copy of the dataframe to avoid altering the original dataframe
    df_test = df.copy()
    
    # Remove specified columns
    if columns_to_remove:
        df_test.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    

    # Remove columns with missing values above the specified threshold
    percent_missing = df_test.isnull().mean() * 100
    columns_to_keep = percent_missing[percent_missing < threshold].index
    df_test = df_test[columns_to_keep]
    
    df_test = select_column_with_max_rows(df_test, 'REF')
    df_test = select_column_with_max_rows(df_test, 'ALT')
   # df_test = select_column_with_max_rows(df_test, 'GT')
        

    # Impute missing values based on specified methods or default method
    for column in df_test.columns:
        if pd.api.types.is_numeric_dtype(df_test[column]):
            missing_percentage = df_test[column].isnull().mean() * 100
            if missing_percentage < 5:
                if (imputation_methods is None) or (column not in imputation_methods):
                    # Median imputation with random deviation
                    median = df_test[column].median()
                    std = df_test[column].std()
                    random_deviation = np.random.uniform(-std, std, size=df_test[column].isnull().sum())
                    df_test.loc[df_test[column].isnull(), column] = median + random_deviation
                    continue

        if imputation_methods and column in imputation_methods:
            impute_info = imputation_methods[column]
            impute_method = impute_info['method']
            impute_type = impute_info['type']

            if impute_type == 'string':
                df_test[column].fillna(impute_method, inplace=True)
            elif impute_type in ['float', 'numeric']:
                df_test[column] = pd.to_numeric(df_test[column], errors='coerce')
                if impute_method == 'max':
                    df_test[column].fillna(df_test[column].max(), inplace=True)
                elif impute_method == 'min':
                    df_test[column].fillna(df_test[column].min(), inplace=True)
                elif impute_method == 'mean':
                    df_test[column].fillna(df_test[column].mean(), inplace=True)
                elif impute_method == 'median':
                    df_test[column].fillna(df_test[column].median(), inplace=True)
                else:  # Assuming impute_method is a specific number
                    df_test[column].fillna(float(impute_method), inplace=True)

    # Check for remaining missing data and report
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0]
    if missing_columns.empty:
        print("All data successfully imputed.")
    else:
        print("Columns with remaining missing data:")
        for column, missing_percent in missing_columns.items():
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")

    return df_test


def data_imputation4(df, threshold_missing_data, imputation_methods=None, columns_to_remove=None):
    '''
    Performs data clean up and imputation on a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold_missing_data (float): Threshold for missing data in percent. Columns with a higher percentage of missing data will be removed.
    - imputation_methods (dict, optional): A dictionary specifying imputation methods for specific columns. The dictionary format should be:
        {'column_name': {'method': 'mean/median/max/min/specific_value', 'type': 'numeric/string'}}
      If no method is specified for a column with less than 5% missing data, random imputation from min/max/mean/median with a standard error is used.

    Returns:
    - DataFrame: The DataFrame after imputation.
    '''

    # Create a copy of the dataframe to avoid altering the original dataframe
    df_test = df.copy()
    
    # Remove specified columns
    if columns_to_remove:
        df_test.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    
    # Remove columns with missing values above the specified threshold
    percent_missing = df_test.isnull().mean() * 100
    columns_to_keep = percent_missing[percent_missing < threshold_missing_data].index
    df_test = df_test[columns_to_keep]

    # Impute missing values based on specified methods or default method
    for column in df_test.columns:
        if pd.api.types.is_numeric_dtype(df_test[column]):
            missing_percentage = df_test[column].isnull().mean() * 100
            if missing_percentage < 5:
                if (imputation_methods is None) or (column not in imputation_methods):
                    # Random imputation from min/max/mean/median with a standard error
                    min_val = df_test[column].min()
                    max_val = df_test[column].max()
                    mean_val = df_test[column].mean()
                    median_val = df_test[column].median()
                    std = df_test[column].std()

                    random_choices = np.random.choice([min_val, max_val, mean_val, median_val], size=df_test[column].isnull().sum())
                    random_deviation = np.random.uniform(-std, std, size=df_test[column].isnull().sum())
                    df_test.loc[df_test[column].isnull(), column] = random_choices + random_deviation
                    continue

        if imputation_methods and column in imputation_methods:
            impute_info = imputation_methods[column]
            impute_method = impute_info['method']
            impute_type = impute_info['type']

            if impute_type == 'string':
                df_test[column].fillna(impute_method, inplace=True)
            elif impute_type in ['float', 'numeric']:
                df_test[column] = pd.to_numeric(df_test[column], errors='coerce')
                # Adjusted code for random imputation with standard deviation
                std = df_test[column].std()
                if impute_method == 'max':
                    random_deviation = np.random.uniform(-std, std)
                    df_test[column].fillna(df_test[column].max() + random_deviation, inplace=True)
                elif impute_method == 'min':
                    random_deviation = np.random.uniform(-std, std)
                    df_test[column].fillna(df_test[column].min() + random_deviation, inplace=True)
                elif impute_method == 'mean':
                    random_deviation = np.random.uniform(-std, std)
                    df_test[column].fillna(df_test[column].mean() + random_deviation, inplace=True)
                elif impute_method == 'median':
                    random_deviation = np.random.uniform(-std, std)
                    df_test[column].fillna(df_test[column].median() + random_deviation, inplace=True)
                else:  # Assuming impute_method is a specific number
                    df_test[column].fillna(float(impute_method), inplace=True)

    # Check for remaining missing data and report
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0]
    if missing_columns.empty:
        print("All data successfully imputed.")
    else:
        print("Columns with remaining missing data:")
        for column, missing_percent in missing_columns.items():
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")

    return df_test

# Note: To use this function, provide a DataFrame 'df' and the necessary parameters.



### Impute data types manually, e.g. P-values impute by maximum or 1.0

#imputation_methods = {'PVAL': 'max'}

imputation_methods = {
    'PVAL': {'method': '1.0', 'type': 'float'},
    'PV4_1': {'method': '1.0', 'type': 'float'},
    'PV4_2': {'method': 'max', 'type': 'numeric'},
    'PV4_3': {'method': 'max', 'type': 'numeric'},
    'PV4_4': {'method': 'max', 'type': 'numeric'}
}

columns_to_remove = ['FREQ', 'GT', 'GT_y']

### Now leave largest REF, ALT and GT fields, remove data that has more than 99% missing values. 
### For values that miss less than 5% imputation using median with 1 std dev for numeric data.

imputed_part1=data_imputation3(df_1,  threshold_missing_data=99, imputation_methods=imputation_methods, columns_to_remove=columns_to_remove)




print("Data Types of All Columns:")

print(imputed_part1.dtypes)
print(imputed_part1.head())

imputation_path="C:/Users/mmiskinyte/Documents/Python_ML/imputation_tests"
imputed_part1.to_csv(f'{imputation_path}/imputed_data_part1.csv')

""" First, we can add ground truth data in our dataset. For true positives choose 1, for others choose 0) """

### Define paths to files and load files
os.environ["ML_GT_FOLDER_PATH"] = "C:/Users/mmiskinyte/Documents/Python_ML/GT"
GT_folder_path = os.environ.get("ML_GT_FOLDER_PATH")
GT_file_path = os.path.join(GT_folder_path, "GT_filtered_for_ML.csv")
filtered_GT_ML = pd.read_csv(GT_file_path, delimiter=';')

print(filtered_GT_ML.dtypes)
print(filtered_GT_ML.head())



def match_groudtruth(dataset, gt_dataset):
    """ Return dataset with ground truth data (1 for variant and 0 for false positive)
    based on the provided gt_dataset. Return dataset only with continuous features. This function
    is to be used for creating dataset for training with ML classifiers.
    """
    # Ensure the POS columns have the same name in both DataFrames or adjust accordingly
    gt_pos_column = 'POS'  # Adjust this if the column name in gt_dataset is different

    # Add a new column 'GROUND' that indicates if the POS is in the ground truth dataset
    dataset['GROUND'] = dataset['POS'].isin(gt_dataset[gt_pos_column]).astype(int)

    # Select only continuous features, assuming they are of type 'float' or 'int64'
    continuous_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Make sure to include the 'GROUND' column in the list if it's not already there
    continuous_columns.append('GROUND') if 'GROUND' not in continuous_columns else continuous_columns

    # Select only the continuous features and the new 'GROUND' column
    dataset3 = dataset[continuous_columns]
    
    # Set 'POS' as the index for dataset3
    dataset4 = dataset3.set_index('POS')

    return dataset4




""" Now we can impute data using more advanced algorithms,
 required for a large chunk of data that is missing and for data that is probably missing not at random """
 

"""Simple - Imputed all data my random median"""
 

def data_imputation_median(df, threshold_missing_data, columns_to_remove=None):
    '''
    Performs data clean up and imputation on a DataFrame using median imputation with random deviation.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold_missing_data (float): Threshold for missing data in percent. Columns with a higher percentage of missing data will be removed.
    - columns_to_remove (list, optional): List of column names to be removed from the DataFrame.
    '''

    # Create a copy of the dataframe to avoid altering the original dataframe
    df_test = df.copy()
    
    # Remove specified columns
    if columns_to_remove:
        df_test.drop(columns=columns_to_remove, errors='ignore', inplace=True)

    # Remove columns with missing values above the specified threshold
    percent_missing = df_test.isnull().mean() * 100
    columns_to_keep = percent_missing[percent_missing < threshold_missing_data].index
    df_test = df_test[columns_to_keep]

    # Impute missing values with median imputation with random deviation
    for column in df_test.columns:
        if pd.api.types.is_numeric_dtype(df_test[column]):
            missing_percentage = df_test[column].isnull().mean() * 100
            if missing_percentage > 0:  # Changed to impute all missing numeric data, regardless of percentage
                # Median imputation with random deviation
                median = df_test[column].median()
                std = df_test[column].std()
                # Impute only the missing values with random values from the normal distribution
                df_test.loc[df_test[column].isnull(), column] = np.random.normal(median, std, size=df_test[column].isnull().sum())

    # Check for remaining missing data and report
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0]
    if missing_columns.empty:
        print("All data successfully imputed.")
    else:
        print("Columns with remaining missing data:")
        for column, missing_percent in missing_columns.items():
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")

    return df_test

 
 
 

"""Data imputation based on knowledge of fields

SGB 

This metric is derived from a Bayesian framework and is used to assess the confidence in the genotyping call made for a variant. The SGB value helps distinguish true genetic variants from sequencing errors or artifacts.
SGB utilizes a Bayesian approach to calculate the probability of a genotype given the observed data. This method takes into account the likelihood of observing the data under different genotype hypotheses.
Positive SGB Values: These generally indicate a high level of confidence in the genotype call. The data strongly support the called genotype.
Negative SGB Values: These suggest lower confidence. The data do not strongly support the called genotype, and there might be a higher likelihood of the variant being a false positive or an artifact.



MQBZ

The Mann-Whitney U test is a non-parametric test used to compare two independent samples to determine whether there is a difference in their distributions. In this case, it's comparing the distributions of mapping quality and base quality for a particular variant.
The Z-score represents the number of standard deviations an element is from the mean of a distribution. In the context of MQBZ:
A positive Z-score indicates that the variant's qualities (MQ and BQ) are higher than the average of all qualities in the dataset, suggesting that the variant is more likely to be a true variant.
A negative Z-score suggests that the variant's qualities are lower than the average, which might indicate a potential false positive, possibly due to sequencing errors or other artifacts.

MQSBZ

A high positive Z-score would indicate that the variant's qualities (MQ and lack of strand bias) are significantly better than the average, suggesting the variant is more likely to be a true variant.
Conversely, a negative Z-score might indicate potential issues with the variant's quality, possibly due to sequencing errors or artifacts.


VDB

The VDB (Variant Distance Bias) metric is used in the context of genomic variant calling. 
It's a statistical measure that assesses the distribution of variant alleles within the reads. 
Specifically, it evaluates whether the variant alleles are randomly distributed along the length of the reads or if they are clustered in a particular region, which might indicate a technical artifact rather than a true genetic variant.
A high VDB score generally indicates a random distribution of the variant alleles across the reads, which supports the authenticity of the variant.
A low VDB score suggests a non-random distribution, raising suspicion about the variant being an artifact of sequencing or alignment errors.


RPBZ

A high positive RPBZ score indicates that the variant alleles are not randomly distributed 
across the read lengths, suggesting a potential bias. 
This might raise concerns about the authenticity of the variant.
A low or near-zero RPBZ score suggests that there is no significant read position bias, 
supporting the validity of the variant call.

BQBZ

"BQBZ" in the context of genomic variant calling stands for "Z-score from the Mann-Whitney U test of Base Quality Bias.
This metric is used to evaluate the quality of variant calls by assessing the distribution of base quality scores at variant positions. 
A high BQBZ score could raise concerns about the reliability of the variant call, as it suggests a significant base quality bias.
A low or near-zero BQBZ score indicates no significant bias, suggesting more confidence in the variant call's accuracy.


ADP

Depth of coverage

GQ

Genotype quality

RD

Reference depth

AD

Allele depth

RBQ - reference based quality

RDF- reference based forward
RDR - reference based reverse


ABQ- base quality of alternative alleles


 """



imputation_methods_knowledge = {
    'SGB': {'method': 'min', 'type': 'float'},
    'MQBZ': {'method': 'min', 'type': 'float'},
    'MQSBZ': {'method': 'min', 'type': 'numeric'},
    'VDB': {'method': 'min', 'type': 'float'},
    'RPBZ': {'method': 'max', 'type': 'float'},
    'BQBZ': {'method': 'min', 'type': 'float'},
    'ADP': {'method': 'median', 'type': 'float'},
    'GQ': {'method': 'min', 'type': 'float'},
    'RD': {'method': 'min', 'type': 'float'},
    'AD': {'method': 'min', 'type': 'float'},
    'RBQ': {'method': 'median', 'type': 'float'},
    'RDF': {'method': 'median', 'type': 'float'},
    'RDR': {'method': 'median', 'type': 'float'},
    'ADF': {'method': 'median', 'type': 'float'},
    'ADR': {'method': 'median', 'type': 'float'},
    'ABQ': {'method': 'median', 'type': 'float'}
    
}



#print(info_fields.keys())


# duplicated_column_pairs = []

# #Iterate over each combination of columns
# for col1, col2 in combinations(df_filtered.columns, 2):
#     if df_filtered[col1].equals(df_filtered[col2]):
#         # If the columns have the same values, add the pair of column names to the list
#         duplicated_column_pairs.append((col1, col2))

# # Check if we found any duplicated columns
# if duplicated_column_pairs:
#     print("Duplicated columns found:")
#     for col1, col2 in duplicated_column_pairs:
#         print(f"Column: {col1} is a duplicate of Column: {col2}")
# else:
#     print("No duplicated columns found.")
















