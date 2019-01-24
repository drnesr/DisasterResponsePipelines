import sys
# From ETL Pipeline Preparation
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads two datasets from the two given paths:
    {messages_filepath}, {categories_filepath}
    and 
    Returns a merged dataframe
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # return a merged dataset
    return pd.merge(messages, categories,how='left', on='id')
    # pass


def clean_data(df):
    '''
    Cleaning the given {df} dataframe
    Returns: A clean dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    category_colnames = [mes[:-2] for mes in row.tolist()]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    return df[~df.duplicated()]
    # pass


def save_data(df, database_filename):
    '''
    Saving the given {df} dataframe as a table in the
    SQL database named {database_filename}
    Returns None
    '''
    # Create the engine using SQLAlchemy
    if database_filename[-3:]!='.db':
        database_filename = database_filename + '.db'
    engine = create_engine(f'sqlite:///{database_filename}')
    # loading the dataframe as a table in the database
    df.to_sql('CategorizedMessages', engine, index=False)
    pass  


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