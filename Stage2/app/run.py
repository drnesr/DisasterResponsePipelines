import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Data, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CategorizedMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # A new visual (Stacked bars for keywords)
    titles_dist = df.groupby('genre').sum()[list(df)[5:]].T
    titles_x = titles_dist.index.tolist()
    # Capitalizing the keywords
    titles_x = [w.replace('_', ' ').capitalize() for w in titles_x]
    y_direct = titles_dist.direct.tolist()
    y_news = titles_dist.news.tolist()
    y_social = titles_dist.social.tolist()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    trace1 = {
    "x": titles_x, 
    "y": y_direct, 
    "marker": {"color": "rgb(141, 99, 35)"}, 
    "name": "Direct", 
    "orientation": "v", 
    "type": "bar"}
    trace2 = {
    "x": titles_x, 
    "y": y_news, 
    "name": "News", 
    "orientation": "v", 
    "type": "bar"}
    trace3 = {
    "x": titles_x, 
    "y": y_social, 
    "marker": {"color": "rgb(243, 219, 32)"}, 
    "name": "Social", 
    "orientation": "v", 
    "type": "bar"}
    data_nesr = Data([trace1, trace2, trace3])
    layout_nesr = {
      "autosize": False, 
      "barmode": "relative", 
      "barnorm": "", 
      "font": {"family": "Roboto"}, 
      "height": 700, 
      "legend": {
        "x": 0.98, 
        "y": 0.98, 
        "borderwidth": 1, 
        "xanchor": "auto"
      }, 
      "margin": {"t": 100}, 
      "plot_bgcolor": "rgb(255, 255, 238)", 
      "title": {
        "x": 0.04, 
        "font": {
          "family": "Roboto", 
          "size": 26
        }, 
        "text": "<b>Distribution of Messages Keywords</b>"
      }, 
      "width": 1200, 
      "xaxis": {
        "autorange": False, 
        "linecolor": "rgb(187, 87, 43)", 
        "linewidth": 5, 
        "range": [-0.5, 34.5], 
        "rangeslider": {
          "autorange": True, 
          "range": [-0.5, 34.5], 
          "visible": False
        }, 
        "showline": True, 
        "showspikes": False, 
        "title": {
          "text": "<b>Messages Keywords&nbsp;</b>", 
          "font": {"size": 17}
        }, 
        "type": "category"
      }, 
      "yaxis": {
        "autorange": True, 
        "gridwidth": 2, 
        "range": [0, 11431.5789474], 
        "showgrid": True, 
        "title": {
          "text": "<b>Count</b>", 
          "font": {"size": 16}
        }, 
        "type": "linear"
      }
    }
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': data_nesr,
            'layout': layout_nesr
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()