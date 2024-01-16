import pandas as pd
import pickle
import argparse
from prediction import preprocessor, predictor_harness

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)

args = parser.parse_args()

input_csv = args.input_csv
output_csv = args.output_csv

df = pd.read_csv(input_csv)

model = load_model()

columns = list(df.columns)

output_csv = predictor_harness(df, model, preprocessor, output_csv, 
                               preproc_params = {'asst_tot_median': 3458428.5, 
                                                 'days_rec_median': 133.75859807872905, 
                                                 'roa_median': 2.39, 
                                                 'debt_ratio_median': 0.5506120856551856, 
                                                 'cash_return_assets_median': 0.021743523917807147, 
                                                 'leverage_median': 0.7460792711253206, 
                                                 'cash_ratio_median': 0.0795322078590196
                                                 })
                               
print('done')

# python3 harness.py --input_csv  <input file in csv> --output_csv <output csv file path to which the predictions are written> 