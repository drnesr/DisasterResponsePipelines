# import packages
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
import warnings

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

# Loading nltk wordsets
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    # Build SQL engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    # Define SQL statement
    sql = 'SELECT * FROM CategorizedMessages'
    df = pd.read_sql(sql, engine)
    X = df.message
    Y = df.iloc[:, 4:]
    # Get Y labels
    Y_labels = list(Y)
    return X, Y, Y_labels


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # Normalize text
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # lemmatize and Remove stop words
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    # Return
    return tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer(sublinear_tf=False)),
                    ('clf',
                    MultiOutputClassifier(
                        RandomForestClassifier(n_jobs=1, 
                        n_estimators=100, 
                        random_state=179,
                        criterion='entropy',
                        max_depth=3,
                        max_features=0.3,
                        min_samples_split=3)))])
    
    parameters = {
    'tfidf__sublinear_tf': (True, False),
    'clf__estimator__min_samples_split': (3, 4),
    'clf__estimator__max_features': ('sqrt', 0.3, 0.4),
    'clf__estimator__max_depth': (3, 5),
    'clf__estimator__criterion': ('gini','entropy'),
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose= 1,n_jobs =-1)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names):

    # Getting the model's predictions
    y_pred = model.predict(X_test)

    # Scoring the outputs
    accuracy = [[(y_pred[:, i] == y_test.values[:, i]).mean(),
                *precision_recall_fscore_support(
                    y_test.values[:, i],
                    y_pred[:, i],
                    average='weighted',
                    labels=np.unique(y_pred[:, i]))]
                for i in range(y_pred.shape[1])]
    accuracy = np.array(accuracy)[:, :-1]
    accuracy = (accuracy * 10000).astype(int) / 100

    print('Showing scores...')
    print('\nAverage scores for all indicators...')
    scores = pd.DataFrame(
        data=accuracy,
        index=category_names,
        columns=['Accuracy %', 'Precision %', 'Recall %', 'F-score %'])
    print(scores.mean(axis=0))
    print('\Detailed scores for each indicator...')
    print(scores)
    return scores


def save_model(model, model_filepath='NesrFittedModel'):
    '''
    Saves the {model} as {model_filepath}
    in pickle format
    Returns:
    None
    '''
    filename=model_filepath
    ## Checking if a File Exists
    #if os.path.isfile(f'./{filename}.sav'):
    #    n = 0
    #    while os.path.isfile(f'{filename}{n:02d}.sav'):
    #        n += 1
    #    else:
    #        filename = f'{filename}{n:02d}.sav'

    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
    print(f'The model has been saved as: {filename}')
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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