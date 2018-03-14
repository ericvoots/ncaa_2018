import pandas as pd

test_df = pd.read_csv('input\\example.csv', delimiter=';')

print(test_df)

out_df = pd.DataFrame()
out_df = test_df.groupby(['ID','Year']).mean().add_suffix('_avg').reset_index()
#out_df['Value2_avg'] = test_df['Value2'].groupby([test_df['ID'], test_df['Year']]).mean()
print(out_df)

print(out_df.columns)