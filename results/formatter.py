import pandas as pd

""" This script formats the csv files into the expected format by Iberlef2025. It takes 3 csv and join all of them"""
if __name__ == '__main__':

    polarity_df = pd.read_csv("Rest-Mex_2025_test_results_Polarity_prompting.csv")
    pueblo_df = pd.read_csv("Rest-Mex_2025_test_results_Town_prompting.csv")
    tipo_df = pd.read_csv("Rest-Mex_2025_test_results_types.csv")

    combined_df = polarity_df.merge(pueblo_df, on="ID", suffixes=('', '_pueblo'))
    combined_df = combined_df.merge(tipo_df, on="ID", suffixes=('', '_tipo'))

    combined_df.columns = ['ID', 'polarity', 'pueblo', 'tipo']

    with open("submission_2.txt", "w", encoding='utf-8') as f:
        for idx, row in combined_df.iterrows():
            line = f'rest-mex\t{row["ID"]}\t{row["polarity"]}\t{row["pueblo"]}\t{row["tipo"]}\n'
            f.write(line)
