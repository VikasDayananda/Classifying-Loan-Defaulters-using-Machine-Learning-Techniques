'''
NaiveBayes Predictor for Loan data.
Project by-
Sharath Kumar
Varun
Vikas Dayananda - 800969865
'''

from __future__ import print_function

import sys	
import os.path			
import numpy as np 
import unicodedata
import pandas as pd
import csv
from operator import add
from pyspark import SparkContext


count_c = dict()
prob_c = dict()
totalRows = 0
delim = '###'
tdict = dict()
variables = dict()
variables['term'] = 0
variables['grade'] = 1
variables['emp_length']=2
variables['home_ownership'] = 3
variables['verification_status'] = 4
variables['loan_status'] = 5
variables['initial_list_status'] = 6

def classifier(data):

    columns = data.split(',')
    target = columns[variables['loan_status']]
    print ('*************************************', target)
    var_prob = []
    
    for variable in variables:
        var_prob.append((variable+'='+columns[variables[variable]],1))
        var_prob.append((variable+'='+columns[variables[variable]]+delim+str(target),1))
    return var_prob

def extract(data):
	columns = data.split(',')
	target = columns[variables['loan_status']]
	return target
def count(data):
	return 1
def groupTuples(tuples):
    mydict = dict()

    for tuple in tuples:
		print("i=", tuple)
		part = tuple[0].split('=')
		print("i[0]", tuple[0])
		print("part", part)
		if  part[0] not in mydict:
			print("part[0]",part[0])
			mydict[part[0]] = dict()
		print("part[1]", part[1])
		values = part[1].split(delim)
		
		if values[0] not in mydict[part[0]]:
			mydict[part[0]][values[0]] = dict()

		if len(values)==1:
			values.append('total')
		mydict[part[0]][values[0]][values[1]] = tuple[1]

    return mydict

def cleanProb(dict_orig):
    
    dict0 = dict_orig.copy()
    targetLevels = len(count_c)
    
    counter = 0    
    for item in dict0:
       print ('item ',item)
       for val in dict0[item]:
			print ('val' ,val)
			for tval in count_c:
				print ('tval' ,tval)
				if tval not in dict0[item][val]:
					dict0[item][val][tval] = 0 
					counter += 1
					print(counter)

    for item in dict0:
        for val in dict0[item]:
            for tval in dict0[item][val]:
                if tval == 'total':
                    dict0[item][val][tval] += (counter*targetLevels)
                else:
                    dict0[item][val][tval] += counter
    print (dict_orig)
    print (dict0)
    return dict0

def predict(row):
    columns = row.split(',')
    
    probs = dict()
    for val in prob_c:
        probs[val] = 1
    
    for var in variables:
        if var == 'loan_status':
            for tval in prob_c:
                probs[tval] *= prob_c[tval]
        else:            
            for tval in prob_c:
                probs[tval] *= pdict[var][columns[variables[var]]][tval]
    
  
    denominator = 0
    for item in probs:
        denominator +=  probs[item]
    
    greatestClass = -1
    greatestVal = 0
    
    for item in probs:
        probs[item] /= denominator
        if probs[item] > greatestVal:
            greatestVal = probs[item]
            greatestClass = item

    return greatestClass
    
if __name__ == "__main__":
   
	#Our data is divided into two parts. Train file and Test file. Train file is used to build our model and test file
	#is for prediciting and verifying.
	#Store path of the training data.
    trainFile = sys.argv[1]
    sc = SparkContext(appName="NaiveBayesP")
    print ('\n')
    print ('\n')
    print("Calling Classifier")
    print ('\n')
    print ('\n')
    rows = sc.textFile(trainFile).flatMap(lambda x: classifier(x)).reduceByKey(lambda val1,val2: val1+val2)
    countRows = sc.textFile(trainFile).map(lambda x: count(x))
    totalRows=sum(countRows.collect())
    
	#Group the processed rows based on Varaiables ( Columns in the data )
    groupedRows = groupTuples(rows.collect())
    print ('\n')
    print ('\n')
    print (rows.collect())
    print ('\n')
    print ('\n')
    print ('\n')
    print (groupedRows)
    print ('\n')
    print ('\n')
    print ('\n')
    
    
    for item in groupedRows['loan_status']:
        count_c[item] = groupedRows['loan_status'][item][item]
		
   
    print ('Total Yes count', count_c['Yes'])
    print ('Total NO count', count_c['No'])
    
    for item in count_c:
        prob_c[item] = float(count_c[item])/float(totalRows)
	
    print ('Prob Yes is', prob_c['Yes'])
    print ('Prob NO is', prob_c['No'])
    
    cleanedgroupedRows = cleanProb(groupedRows)
         
    #print(count_c)
    #print(prob_c)
    
    pdict = cleanedgroupedRows.copy()
    cond_probs = dict()
    #print(pdict)
    

    #Calculate conditional probabilities.
    for item in pdict:
        for val in pdict[item]:
            for entry in pdict[item][val]:
                if entry not in 'total':
                    pdict[item][val][entry] /= float(count_c[entry])
    print(pdict)
    
    print('Training Complete!!!!!!!')
    print ('Rows=',totalRows1)

    testFile = sys.argv[2]
    results = sc.textFile(sys.argv[2]).map(lambda x: predict(x))
    pred_col = sc.textFile(sys.argv[2]).map(lambda x: extract(x))
    print('Predicitons Complete!!!!!')
   
    output = results.collect()
    approval=[]
    #with open('/users/vdayanan/input/test.csv') as csvDataFile:
	#	csvReader = csv.reader(csvDataFile)
	#	for row in csvReader:
	#		approval.append(row[5])
    
    approval=pred_col.collect()
    
    
    print('==================================================================================================================')
    print('Results:')
    print (sum(1 for a, b in zip(approval, output) if a == b),' of ', len(approval),' are correct')
    print ('Naive Bayes Model Accuracy: ',end='')
    accuracy=float(sum(1 for a, b in zip(approval, output) if a == b))/len(approval)*100
    print(accuracy,'%')
    print('==================================================================================================================')
    sc.stop()