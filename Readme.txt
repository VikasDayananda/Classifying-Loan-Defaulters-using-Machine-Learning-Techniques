Technical Requirements:
1) Python 3.5 or above
2) Cloudera VM
3) Hadoop installed on the VM
4) Apache Spark installed on the VM
5) Access to DSBA cluster
6) Jupyter Notebook with PySpark integrated (optional)

Below, I am mentioning the steps to follow to run the program on the DSBA cluster:

Step 1: Place the input csv files onto the cluster server.

Step 2: Place the .py code files onto the cluster server.

Step 3: Transfer all the above mentioned files to hdfs.

	$ hadoop fs -put <file from the cluster> <new hdfs path>

Step 4: Once the files are placed on hdfs, execute the programs one by one as follows:
		
	$ spark-submit NaiveBayes.py <training file path> <test file path>
	$ spark-submit kNearest.py k <training file path> <test file path> //K is desired number of neighbours



Files in the submission folder:

kNearest.py: code file for implementation of K-Nearest Neighbor algorithm

NaiveBayes.py: code file for implementation of Naive Bayes algorithm

train.csv: csv file consisting of the training data

test.csv: csv file consisting of the test data

OPTIONAL: Running the program on Jupyter Notebook (execution time is more)

Step 1: Place all the .py files and the csv files into your working directory

Step 2: Open the necessary .py file and execute it either module by module to check the output at every stage or run the complete code at once by pressing Shift+Enter
 