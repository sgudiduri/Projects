import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("housingDataset.csv")
df.head()
# print(df['Neighborhood'].unique())
profile = ProfileReport(df, missing_diagrams=None, duplicates=None, interactions=None)
profile.to_file("result_housing_combined.html")