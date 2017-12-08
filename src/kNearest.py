# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:02:51 2017

@author: Sharath
"""

from pyspark import SparkContext
import sys
import math
import pandas as pd

# list to contain training data

trainData = []

# state the value of 'K' as an argument

k = 3

# indicate the predictor variable

targetVariable = 'loan_status'

# define all the independent variables chosen for classification

# take into a dictionary

dataVariable = dict()

# specify the column numbers of all the variables in the data set

# so we have a dictionary of the form "key: column_name", "value: column_number"

dataVariable['loan_amnt'] = 7
dataVariable['int_rate'] = 8
dataVariable['annual_inc'] = 9 
dataVariable['dti'] = 10
dataVariable['open_acc'] = 11
dataVariable['revol_bal'] = 12
dataVariable['total_acc'] = 13
dataVariable['revol_util'] = 14
dataVariable['delinq_amnt'] = 15
dataVariable['loan_status'] = 5
            
def train(data):
    rows = str(data).split(',')
    val = dict()
    for d in dataVariable:
        if d != targetVariable:
            val[d] = float(rows[dataVariable[d]])
        else:
            val[d] = rows[dataVariable[d]]

    return val

def predictTarget(data):
    rows = str(data[0]).split(',')
    distance = []
    
    for td in trainData:
        dist = 0
        for key in td:
            if key != targetVariable:
                
    # find the euclidean distances between the points and sort them in decreasing order

                dist += (td[key] - float(rows[dataVariable[key]])) ** 2

        distance.append((math.sqrt(dist), td[key]))

    ordered_dist = sorted(distance, key=lambda x: x[0])[0:k]

    
    out = dict()
    
	for d in ordered_dist:
		if d[1] not in out:
			out[d[1]] = 0
		out[d[1]] += 1
        
    gk = -1
    gv = 0
    for i in out:
        if out[i] > gv:
            gk = i
			gv = out[i]
    
    return gk

if __name__ == "__main__":

    # pass the training and test files as arguments
    trainingFile = sys.argv[1]
    testFile = sys.argv[2]
    
    # initiate a SparkContext
    sc = SparkContext(appName="KNN")
    
    #Read in the training data and store it in RDD
    trainRead = sc.textFile(trainingFile).map(lambda x: train(x))
    trainData = trainRead.collect()
    
    print('#################### Training Step Completed ####################')
    
    # Read in the test files and get the values of the ouptut and store it in RDD
    testRead = sc.textFile(testFile).map(lambda x: predictTarget([x]))   
    pred_status = testRead.collect()
    
    print('#################### Loan_status Prediction Completed ####################')
    
    # read the test file using pandas
    t = pd.read_csv(testFile,header=None)

    # Select only the loan_status column
    actual_status = t[5]

    print ('KNN Model Accuracy: ',end='')

    # compare the actual loan status with that of the predicted one
    print((actual_status == pred_status).sum()/t.shape[0])

    # Stop the SparkContext
    sc.stop()