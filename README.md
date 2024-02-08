# About project
This is an implementation of a random forest algorithm. It consists of two types of classifiers which are NBC and ID3. Core of the ID3 is based on existing implementation but it was slightly changed in order to fit project requirements. Project was conducted in pair.

# Selecting Research/Experiments to run:
In order to conduct experiments configure main.py file by providing proper experiment methods. For each experiment, input a string corresponding to the respective dataset in the argument. 
Experiment results are stored in proper catalogues in the form of .csv tables, images presenting diagrams and confusion matrices.

The available options are:

    'corona' -> corona.csv
    'divorce' -> divorce.csv
    'glass' -> glass.csv
    'letter' -> letter-recognition.csv
    'loan_approval' -> loan_approval.csv

Datasets where downloaded from the kaggle.

# Required modules:
    scikit-learn
    pandas
    matplotlib
    seaborn
    openpyxl

# Running program from the /code directory:
    python main.py
