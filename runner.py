import pandas as pd
import os
import glob
import sys
import time
import numpy as np
from model import Trainer

PATH = '../CICFlowMeter-4.0/bin/data/daily/'

target_file = sorted(glob.glob(PATH + '*.csv'))[-1]


separator = ','
reader = open(target_file, 'r')
header = reader.readline().split(separator)
count = 0

print('Number of columns: %d' % len(header))
print('Reading %s\n' % target_file)

model = Trainer()
model.load_model('./SVM_classifier.sav')

while True:
    row = reader.readline()
    if not row:
        time.sleep(0.1)
        continue
    count += 1
    # sys.stdout.write('\r' + str(count))
    # sys.stdout.flush()

    # Preprocess
    row = row.split(separator)[:-1]
    row = np.delete(np.array(row), [0, 1, 2, 3, 5, 6], 0)
    row = row.astype(np.float32)

    # Classify
    label = model.predict([row])
    print(label)
