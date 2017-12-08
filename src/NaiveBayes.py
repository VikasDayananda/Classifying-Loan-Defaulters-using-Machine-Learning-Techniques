'''
NaiveBayes Predictor for Loan data.
Project by-
Sharath Kumar 	- 800975820
Varun Rao		-800959522
Vikas Dayananda -800969865
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

#store decision counts
decision_count = dict()

#store decision counts
decision_prob = dict()

totalRows = 0
DELIM = '###'
tdict = dict()
prob = dict()

#List attribute names
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
        var_prob.append((variable+'='+columns[variables[variable]]+DELIM+str(target),1))
    return var_prob

def extract(data):
	columns = data.split(',')
	target = columns[variables['loan_status']]
	return target
def count_Yes(data):
    columns = data.split(',')
    target = columns[variables['loan_status']]
    if target=='Yes':
		return 1
    else: 
		return 0
def count_No(data):
    columns = data.split(',')
    target = columns[variables['loan_status']]
    if target=='No':
		return 1
    else: 
		return 0
def count(data):
	return 1
def groupTuples(tuples):
    temp = dict()

    for tuple in tuples:
		print("i=", tuple)
		part = tuple[0].split('=')
		print("i[0]", tuple[0])
		print("part", part)
		if  part[0] not in temp:
			print("part[0]",part[0])
			temp[part[0]] = dict()
		print("part[1]", part[1])
		values = part[1].split(DELIM)
		
		if values[0] not in temp[part[0]]:
			temp[part[0]][values[0]] = dict()

		if len(values)==1:
			values.append('count')
		temp[part[0]][values[0]][values[1]] = tuple[1]

    return temp

def cleanProb(dict_orig):
    
    temp = dict_orig.copy()
    targetLevels = len(decision_count)
    
    counter = 0    
    for item in temp:
       print ('item ',item)
       for val in temp[item]:
			print ('val' ,val)
			for val2 in decision_count:
				print ('val' ,val)
				if val2 not in temp[item][val]:
					temp[item][val][val2] = 0 
					counter += 1
					print(counter)

    for item in temp:
        for val in temp[item]:
            for val2 in temp[item][val]:
                if val == 'count':
                    temp[item][val][val2] += (counter*targetLevels)
                else:
                    temp[item][val][val2] += counter
   
    return temp

def predict(row):
    columns = row.split(',')
    
    
    for val in decision_prob:
        prob[val] = 1
    
    for var in variables:
        if var == 'loan_status':
            for val2 in decision_prob:
                prob[val2] *= decision_prob[val2]
                print ('Prob of ',val2,' is =',prob[val2])
        else:            
            for val2 in decision_prob:
                prob[val2] *= pdict[var][columns[variables[var]]][val2]
                print ('Prob of ',val2,' is =',prob[val2])
    
  
    denominator = 0
    for item in prob:
        denominator +=  prob[item]
    
    greatestClass = -1
    greatesval = 0
    
    for item in prob:
        prob[item] /= denominator
        if prob[item] > greatesval:
            greatesval = prob[item]
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
        decision_count[item] = groupedRows['loan_status'][item][item]
		
   
    print ('Total Yes count', decision_count['Yes'])
    print ('Total NO count', decision_count['No'])
    
    for item in decision_count:
        decision_prob[item] = float(decision_count[item])/float(totalRows)
	
    print ('Prob Yes is', decision_prob['Yes'])
    print ('Prob NO is', decision_prob['No'])
    
    cleanedgroupedRows = cleanProb(groupedRows)
         
    #print(decision_count)
    #print(decision_prob)
    
    pdict = cleanedgroupedRows.copy()
    cond_prob = dict()
    print(pdict)
    

  
    for item in pdict:
        for val in pdict[item]:
            for entry in pdict[item][val]:
                if entry not in 'count':
                    pdict[item][val][entry] /= float(decision_count[entry])
    print(pdict)
    
    print('Training Complete!!!!!!!')
    print ('Rows=',totalRows)

    testFile = sys.argv[2]
    results = sc.textFile(sys.argv[2]).map(lambda x: predict(x))
    total_Yes=sc.textFile(sys.argv[2]).map(lambda x: count_Yes(x))
    total_No=sc.textFile(sys.argv[2]).map(lambda x: count_No(x))
    pred_col = sc.textFile(sys.argv[2]).map(lambda x: extract(x))
    print (prob)
    print('Predicitons Complete!!!!!')
   
    output = results.collect()
    approval=[]
    #with open('/users/vdayanan/input/test.csv') as csvDataFile:
	#	csvReader = csv.reader(csvDataFile)
	#	for row in csvReader:
	#		approval.append(row[5])
    
    approval=pred_col.collect()
    yesCounts=sum(total_Yes.collect())
    noCounts=sum(total_No.collect())
    
    
    print('==================================================================================================================')
    print('Results:')
    print (sum(1 for a, b in zip(approval, output) if a == b),' of ', len(approval),' predictions are correct')
    print (sum(1 for a, b in zip(approval, output) if (a == b and a=='Yes')),' of ',yesCounts,' are True Positives')
    print (sum(1 for a, b in zip(approval, output) if (a == b and a=='No')),' of ',noCounts,' are True Negatives')
    print (sum(1 for a, b in zip(approval, output) if (a == 'Yes' and b=='No')),'are False Negatives')
    print (sum(1 for a, b in zip(approval, output) if (a == 'No' and b=='Yes')),'are False Positives')
    print ('Naive Bayes Model Accuracy: ',end='')
    accuracy=float(sum(1 for a, b in zip(approval, output) if a == b))/len(approval)*100
    print(accuracy,'%')
    print('==================================================================================================================')
    sc.stop()