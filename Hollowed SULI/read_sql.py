
# For help on using sql with Pandas see
# http://www.pererikstrandberg.se/blog/index.cgi?page=PythonDataAnalysisWithSqliteAndPandas

# for help on data analysis with Pandas see
# http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb

import sqlite3
import pandas as pd
import numpy as np

# Create your connection.
print("reading SQL file")
cnx = sqlite3.connect('detroit-Demand.sqlite')

print("extracting data from SQL")
households = pd.read_sql_query("SELECT * FROM Household", cnx)
people = pd.read_sql_query("SELECT * FROM Person", cnx)
locations = pd.read_sql_query("SELECT * FROM All_Locations", cnx)
activity = pd.read_sql_query("SELECT * FROM Activity", cnx)


# for help on working with 

# sum up the values in the column of households called "people"
households.people.sum()
# an alternative formulation
households['people'].sum()
