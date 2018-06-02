#!/usr/bin/python
#
# Example in this file is adapted from,
# https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md


from __future__ import print_function

import numpy as np
import tflearn

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv

data, labels = load_csv('training_data.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# Cleanup data
def preprocess(dataset, columns_to_delete):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [data.pop(column_to_delete) for data in dataset]
    return np.array(dataset, dtype=np.float32)

# Configure coloumns in dataset to ignore
to_ignore=[0]

# Preprocess/cleanup data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 3])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=10, show_metric=True)

# Create some test cases
test_case1 = ['John', 5700, 10008, 28]  # We 'know' that it should be '1'
test_case2 = ['Henry', 3560, 11115, 29]  # We 'know' that it should be '0'

# Preprocess data
test_case1, test_case2 = preprocess([test_case1, test_case2], to_ignore)

# Predict
pred = model.predict([test_case1, test_case2])
print("John's probability to be '1' :", pred[0][1])
print("Henry's probability to be '1':", pred[1][1])
