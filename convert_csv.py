import os
import pandas as pd


aggregate_df = pd.DataFrame()
new_df = pd.DataFrame() 

path = r'./CSVs'
files = os.listdir(path)
# index for limiting how many values in the sample csv. 
index = 0
for file in files:
    if index == 50:
        break
    if file.endswith('.csv'):
        data = pd.read_csv("{}/{}".format(path, file))
        data.pop('Unnamed: 0')
        # data.pop('L7_http')
        # data.pop('L4_tcp')
        # data.pop('L4_udp')
        # data.pop('L7_https')
        # data.pop('port_class_src')
        # data.pop('port_class_dst')
        # data.pop('NTP_count')
        aggregate_df = pd.concat([data, aggregate_df], ignore_index=True)
        # index += 1

aggregate_df.to_csv("combined_dataset.csv")

for column in aggregate_df:
    vals = aggregate_df[column].tolist()
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
new_df.to_csv("normalised_dataset.csv", index=False)