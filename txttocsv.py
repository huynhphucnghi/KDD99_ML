import csv
import numpy as np 

path = 'Data/'
filetxt = 'kddcup.data_10_percent_corrected'
filecsv = 'data.csv'

with open(path + filetxt, 'r') as f:
    stripped = (line.strip() for line in f)
    lines = (line.split(",") for line in stripped if line)
    with open(path + filecsv, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
