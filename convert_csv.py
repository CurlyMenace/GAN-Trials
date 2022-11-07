import pandas as pd
data = pd.read_csv("dataset.csv")
new_df = pd.DataFrame() 

for column in data:
    vals = data[column].tolist()
    maxVal = max(vals)
    minVal = min(vals)

    normalised_data = list()
    for x_val in vals:
        z_val = 0
        if ((maxVal - minVal) > 0 and (x_val - minVal) > 0): 
            z_val = (x_val - minVal) / (maxVal - minVal)
        normalised_data.append(z_val)
    new_df[column] = normalised_data

print(new_df)
new_df.to_csv("normalised_dataset.csv", header=False, index=False)