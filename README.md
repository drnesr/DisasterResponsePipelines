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

### Stage 2. Updating the web app files

In tHis stage, we moved the final code from the jupyter notebooks to the following `.py` files:
1. `Stage2\data\process_data.py` for reading, cleaning, and reloading the data to SQL db.
> The original data is in `disaster_categories.csv` and `disaster_messages.csv` files, and the output file is `DisasterResponse.db`
2. `Stage2\models\train_classifier.py`
> The code reads the db file then exports the trained model to `classifier.pkl` file.
3. `Stage2\app\run.py` which run the web app.

> **To run the web app** <br> 1. From the terminal, `cd app` and then `python run.py` commands. <br> 2. In the browser, type http://localhost:3001/ the app will run
