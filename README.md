# DisasterResponsePipeline
Disaster Response Pipeline created by Aseem Narula for Udacity Data Scientist Nanodegree 

In this GitHub repository, I have done the data analysis using the Pandas and Seaborn, Matplotlib data visualisation libraries in the Pyton.

I have used the Scikit Learn Python libraries to create the ML and ETL pipeline for my Disaster Reponse Project.

The objective of this project is to build using the Machine Learning Pipeline to categorize the emergency messages based on the data set containing real messages that were sent during disaster events.

This project consists of the three components :
a) ETL Pipeline
b) ML Pipeline
c) Flask Web App

Requierments:
You need following python packages: 
* flask
* plotly
* sqlalchemy
* pandas
* numpy
* sklearn
* nltk

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



Screenshots of the web app are added in the Medium blog - 
https://aseemnarula.medium.com/disaster-response-pipeline-machine-learning-project-6dfce162facf

Acknowledgement
All the datasets of Messages and Categories used in this Data Science Project are provided through Figure Eight in collaboration with the Udacity and are used for my project with Udacity Data Scientist Nanodegree.
