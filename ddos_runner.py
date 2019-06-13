import os
import glob
import sys
import time
import numpy as np
from keras.models import load_model
import joblib

PATH = './CICFlowMeter-4.0/bin/data/daily/'

while len(glob.glob(PATH + '*.csv')) == 0:
    pass
target_file = sorted(glob.glob(PATH + '*.csv'))[-1]


separator = ','
reader = open(target_file, 'r')
header = reader.readline().split(separator)
count = 0

print('Number of columns: %d' % len(header))
print('Reading %s\n' % target_file)

model = load_model('models/ddos_model.h5')
scaler = joblib.load('models/ddos_scaler.pkl')

counts = [0, 0]

outputFile = open('mlp.csv', 'w')

while True:
    row = reader.readline()
    if not row:
        time.sleep(0.1)
        continue

    # Preprocess
    row = row.split(separator)[:-1]
    id = row[0]
    row = np.delete(np.array(row), [0, 1, 2, 3, 4, 5, 6, 71], 0)
    # Handle NaN and Infinity
    for i in range(len(row)):
        if row[i] == 'NaN':
            row[i] = 0
        elif row[i] == 'Infinity':
            row[i] = 1.79e+308
    row = row.astype(np.float64)
    # Classify
    label = model.predict_classes(scaler.transform([row]))[0]
    
    # Update & print
    count += 1
    counts[int(label)] += 1
    outputFile.write('%s, %d\n' % (id, label))

    if int(label) != 0:
        sys.stdout.write('\r%-50s -> %10d\n' % (id, label))
    sys.stdout.write('\rCount: %d, DDoS: %d' % (count, counts[1]))
    sys.stdout.flush()
