# Disaster Response Pipelines
A project that uses data engineering skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. The project is a part of [Udacity's](https://www.udacity.com/) [Data Scientist nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

## Project Components and Stages
There are three components for this project in two stages.

**The components are:**
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

### Stage 1. Through Jupyter notebooks

This stage includes working with two jupyter notebook files:
1. `ETL Pipeline Preparation.ipynb` which is used to read, clean, and reload the data both in normal way and through pipelines.
2. `ML Pipeline Preparation.ipynb` which is used to construct a ML algorithm to predict the keywords from messages. The operation also includes both in normal way and through pipelines.

### Stage 2. Updating the web app files (*YOU ARE HERE*)

In tHis stage, we moved the final code from the jupyter notebooks to the following `.py` files:
1. `Stage2\data\process_data.py` for reading, cleaning, and reloading the data to SQL db.
> The original data is in `disaster_categories.csv` and `disaster_messages.csv` files, and the output file is `DisasterResponse.db`
2. `Stage2\models\train_classifier.py`
> The code reads the db file then exports the trained model to `classifier.pkl` file.
3. `Stage2\app\run.py` which run the web app.

### Instructions To build the model and the database:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (**See below for instructions**)

> **To run the web app from local terminal** <br> 1. From the terminal, `cd app` and then `python run.py` commands. <br> 2. In the browser, type http://localhost:3001/ the app will run

> **To run the web app from Udacity Terminal** <br> 1. Run your app with cd app and python run.py commands
<br> 2. Open another terminal and type env|grep WORK this will give you the spaceid it will start with viewXXXXXXX and some characters after that<br> 3. 
Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id you got in the step 2<br> 4. Press enter and the app should now run for you