# Was the movie good enough?
#### COMP 551 [project 2](https://cs.mcgill.ca/~wlh/comp551/files/miniproject2_spec.pdf)

In this project we explore the performance of various classification models for predicting the sentiments of reviews on the popular website IMDb. 

# Install

Install dependencies with conda

`conda create -n newenvironment --file requirements.txt`

# Project structure

The project is structured as follows -
1. **data:** contains the reviews distributed into train and test subfolders.
2. **output** containing the final exported csv, with predictions from our best model.
3. **src** contains the source code of the project, consisting of the following files
    1. *classifiers.py*: Implementation of the various classifiers
    2. *imdb_main.py*: Main file containing menu based options to choose which type of classifier, pipeline the user wants to run
    3. *load_data.py*: Loading of the data from the data folder
    4. *naive_bayes_from_scratch.py (Main file)*: Implementation from sctrach, based on the Lecture Slides.
    5. *pipeline_data.py*: Implementation of pipelines for feature extraction and model validation
    6. *preprocessing_data.py*: Contains all the NLP techniques used by us for preprocessing the data
    7. *vectorization_split.py*: Contains the splitting of data into train and validation datasets
    8. *IMDB_Sentiment_Analysis_Group_51.ipynb*: analyses of the data and execution of models.

# Running


1. Execute models and analyses:

'jupyter notebook .'

2. Detailed Model execution with all experiments

'imdb_main.py'