# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#importing the NLP libraries
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

#importing database engine library
from sqlalchemy import create_engine

#importing the sklearn libraries
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score, make_scorer
import pickle



def load_data(database_filepath):
    
    ''' Input Parameter --- Database FilePath '''
    ''' Desctiption -----> This takes the input database file path location where the cleaned database is stored after ETL cleaning'''
    
 
    ''' Output Parameters --- X, Y, Y.columns '''
    ''' Desctiption -----> This returns the X, Y and the Y.columns as an output parameters'''
    
    # load data from database
    #engine = create_engine('sqlite:///DisasterResponse.db') 
    #engine = create_engine('sqlite:///../data/DisasterResponse.db')
    
    print("Database location--> ",database_filepath)
    print("Full Database location ---->", 'sqlite:///',database_filepath)
    
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Reading data from 'MessagesCategories' table into the Dataframe 
    df = pd.read_sql("SELECT * FROM MessagesCategories", con=engine)
    
    # Asssigning the X and Y variable so that it can be used to test and train the dataset
    
    # Storing message column of the dataframe into the X variable
    X = df.message.values
    
    # Storing the remaining columns into the Y variable
    Y = df.iloc[:,4:]
    
    # Checking if the Y variable is having the Y variable to see if related column has only binary values of 0 and 1 
    print("Checking the Y variable to see if related column has only binary values of 0 and 1 \n")
    print(Y.related.unique())
    
    # Returning the variables X, Y and Y.columns from def - load_data
    return X, Y, Y.columns
    



# A tokenization function to process text data the Disaster Response Messages from the Dataframe 

def tokenize(text):
    
    """Tokenization function."""
    """Input Parameter Description -----> Receives as input as raw text"""
   
    """"Does the following task"""
    """Normalizetext, Stop words removal, Stemming and Lemmatizing."
    """""""Returns tokenized text"""
    
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        #---Inside the For loop and replacing the url place holder---
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
       
        # ---Removing the Stop Words---     
        if tok in stopwords.words("english"):
            continue
            
        # ---Reducing words to their Stems---        
        tok = PorterStemmer().stem(tok)
        
        # ---Reduce words to their root form---        
        tok = lemmatizer.lemmatize(tok).lower().strip()
        
        #---Appending the clean tokens to the list---"
        clean_tokens.append(tok)
        
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    
    return clean_tokens



def build_model():

    ''''This machine pipeline is taking in the message column as input and output classification results on the other 36 categories in the dataset.'''''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    
    # Doing the cross validation with the parameters
    parameters = {'clf__n_estimators': [10],
             'clf__min_samples_split': [2]}   
   
   
    cv = GridSearchCV(pipeline, parameters,verbose = 3,n_jobs=-1)
        
        
    return cv
    
    

def evaluate_model(model, X_test, Y_test,category_names):
    
    '''' Evaulate ML Model '''
    ''' Input Parameters --- model, X_test, Y_test, category_names '''
    
    #Predicting the Y variable
    #Y_pred = pipeline.predict(X_test)
    print("X test values")
    X_test[1:3]
    
    print("Y test values")
    Y_test[1:3]
    
    
    Y_pred = model.predict(X_test)
    
    # calling sklearn's classification_report on each column.
    #for i_value, col in enumerate(Y_test.columns):   
    for i_value, col in enumerate(category_names):
        print(col)
        print("column_name----->",col,classification_report(Y_test[col], Y_pred[:,i_value]))
    
    # Printing the overall Accuracy of the model
    overall_avg_value = (Y_pred == Y_test).mean().mean()
    print("Accuracy Overall:\n", overall_avg_value)
     

def save_model(model, model_filepath):
    
    ''' Saving the model '''
    ''' Input Parameters ----> Model and Model Filepath '''
    ''' Description ----> This will create the pickle file for the classifier '''
    
    pickle.dump(model, open("classifier.pkl", 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        #evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()