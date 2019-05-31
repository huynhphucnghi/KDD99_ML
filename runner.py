import pandas as pd
import os
import glob
import sys
import time
import numpy as np
from keras.models import load_model
import joblib

PATH = '../CICFlowMeter-4.0/bin/data/daily/'

target_file = sorted(glob.glob(PATH + '*.csv'))[-1]


separator = ','
reader = open(target_file, 'r')
header = reader.readline().split(separator)
count = 0

print('Number of columns: %d' % len(header))
print('Reading %s\n' % target_file)

model = load_model('models/ddos_model.h5')
scaler = joblib.load('models/ddos_scaler.pkl')

while True:
    row = reader.readline()
    if not row:
        time.sleep(0.1)
        continue
    count += 1
    sys.stdout.write("\rFlow: " + str(count))
    sys.stdout.flush()

    # Preprocess
    row = row.split(separator)[:-1]
    id = row[0]
    row = np.delete(np.array(row), [0, 1, 2, 3, 4, 5, 6, 71], 0)
    row = row.astype(np.float64)
    # Classify
    label = model.predict_classes(scaler.transform([row]))[0]
    if label > 0:
        print('\r%-50s -> %d' % (id, label))
