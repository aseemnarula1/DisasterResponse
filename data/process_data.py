import sys
import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    
    '''Input Paramter -- Messages File Path, Categories File Path'''
    '''Description -----> This takes a two input parameters messages and categories csv source file paths'''
    
    '''Output Paramter -- A DataFrame with merged datasets'''
    '''Description ------> This return the dataframe containing the Disaster Messages and Categories data'''
    
    # Read messages csv file
    messages = pd.read_csv(messages_filepath)
    
    # Read categories csv file
    categories = pd.read_csv(categories_filepath)
    
    # Merging two dataframe based on the 'id' column 
    df = pd.merge(messages, categories, on="id")
    
    # Returing df output argument
    return df


def clean_data(df):
    
    '''Input Paramter -- DataFrame '''
    '''Description -----> This takes a input parameters as dataframe containing the merged dataset'''
    
    '''Output Paramter -- A DataFrame with merged datasets'''
    '''Description ------> This return the dataframe containing the Disaster Messages and Categories data'''
    
    # Create dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # Select the first row of the categories dataframe
    first_row = categories.iloc[0]
    
    # Looping through the first_row created above to remove the last 2 characters of the category
    category_colnames = [val[:-2] for val in first_row]
    
    # Renaming the columns of the 'categories'
    categories.columns = category_colnames
    
    for column in categories:
        # Setting each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # Converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
    
    # Dropping the original categories column from 'df'
    df.drop("categories", axis=1, inplace=True)
    
    # Concatenating the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    # Checking the number of duplicate rows
    duplicate = df.duplicated(keep="first").sum()  
    print("Number of the Duplicate Rows before removal :",duplicate) 
    
    # Dropping Duplicates
    df.drop(df[df.duplicated(keep="first")].index, inplace=True)
    
    # Checking the number of Duplicate rows after removal
    check_duplicates = df.duplicated(keep="first").sum()
    print("Number of the Duplicate Rows :",check_duplicates) 
    
    #Checking the dataframe statistics - mean, std, min, max, 25%,50%,75% percentile
    df.describe()
    
    # In the above statistics,the column 'related' is showing max value of 2 which is not correct, 
    # checking the dataframe for the values of the related column having value equals to 2.
    df[df['related'] > 1].head(2)
    
    #I have seen that for those rows with 'related' = 2, all other label values are 0. 
    #Hence I conclude that the original label 'related-2' actually means 'related-0'. Then we can correct them:
    # set labels 2 to 0
    df.loc[df['related'] > 1,'related'] = 0
    
    # checking the total entries with wrong labels value of 2 after correction 
    print("checking the total entries with wrong labels value of 2 after correction, this should be zero")
    (df['related'] > 1).sum()    
        
    # Returning cleaned dataframe as 'df' 
    return df



def save_data(df, database_filename):
    
    '''Input Paramter -- DataFrame '''
    '''Description -----> This takes a input parameters as dataframe containing the clean dataframe'''
    
    '''Input Parameter -- Database FileName'''
    '''Description -----> Location where we want to save the DisasterResponse Database for further processing in ML Pipeline'''
    
    
    '''Output Paramter -- None'''
    
    from sqlalchemy import create_engine
    
    # Creating the SQL engine for the Disaster Response database
    #engine = create_engine('sqlite:///DisasterResponse.db') # this is working fine
    
    
    #engine = create_engine('sqlite:///../data/DisasterResponse.db') --yet to test
    
    engine = create_engine('sqlite:///' + database_filename)
    
    # Dataframe will be saved in the 'MessagesCategories' table 
    df.to_sql('MessagesCategories', engine, index=False, if_exists = 'replace')
    
    # Disposing the Engine instance
    engine.dispose()
    
    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()