#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Sample pyspark script to be uploaded to Cloud Storage and run on
Cloud Dataproc.

Note this file is not intended to be run directly, but run inside a PySpark
environment.
"""

# [START pyspark]
# imports
import pyspark
import re
import ast
import time
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from os import path

from pyspark import SparkContext

################################################################################
#Uncomment For GCP only
################################################################################
##10K line file
#testRDDfile = 'gs://vnagapudi-w261-final/train_10k.txt'
##5% RDD file
#sampRDDfile = 'gs://vnagapudi-w261-final/train_sample.txt'
##100% RDD file
#fullRDDfile = 'gs://vnagapudi-w261-final/full_sample.txt'

#Spark context
#sc = pyspark.SparkContext()


##Specify outputs...
################################################################################


################################################################################
#For Local only
################################################################################
testRDDfile = 'data/train_10k.txt'
#5% RDD file
sampRDDfile = 'data/train_sample.txt'
#100% RDD file
#fullRDDfile = 'data/full_sample.txt'

#Spark context
#sc = SparkContext("local", "Final Project")
from pyspark.sql import SparkSession
app_name = "final_project"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext
################################################################################


################################################################################
#README: This is just copied over from the master notebook.
#Please sync up every so often as needed
################################################################################

################################################################################
##########################    All functions here ###############################
################################################################################

################################################################################
def normalize(dataRDD):
################################################################################
    """
    Scale and center data round mean of each feature.
    Args:
        dataRDD - records are tuples of (y, features_array)
    Returns:
        normedRDD - records are tuples of (y, features_array)
    """
    featureMeans = dataRDD.map(lambda x: x[1]).mean()
    featureStdev = np.sqrt(dataRDD.map(lambda x: x[1]).variance())

    normedRDD = dataRDD.map(lambda x: (x[0], (x[1] - featureMeans)/featureStdev))

    return normedRDD

################################################################################
def LogLoss(dataRDD, W, regType = None, regParam=0.05):
################################################################################
    """
    Compute log loss function.
    Args:
        dataRDD - each record is a tuple of (y, features_array)
        W       - (array) model coefficients with bias at index 0
        regType - (str) 'ridge' or 'lasso', defaults to None
        regParam - (float) regularization term coefficient defaults to 0.1
    Returns:
        loss - (float) the regularized loss
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[1]), x[0])).cache()

    # add regularization term
    reg_term = 0
    if regType == 'ridge':
        reg_term = regParam*np.linalg.norm(W[1:])
    elif regType == 'lasso':
        reg_term = regParam*np.sum(np.abs(W[1:]))
        
    #broadcast model
    #W = sc.broadcast(W) #uncomment this line when deploying it on the cloud
    
    # compute loss
    loss = augmentedData.map(lambda x: x[1]*np.log(1 + np.exp(-np.dot(x[0], W))) + \
                             (1 - x[1])*(np.dot(x[0], W) + np.log(1 + np.exp(-np.dot(x[0], W))))).sum()\
                            /augmentedData.count()
    
    return loss

################################################################################
def LogLoss_grad(dataRDD, W, regType = None, regParam=0.05):
################################################################################
    """
    Compute log loss function inside gradient descent.
    Args:
        dataRDD - each record is a tuple of (y, features_array)
        W       - (array) model coefficients with bias at index 0
        regType - (str) 'ridge' or 'lasso', defaults to None
        regParam - (float) regularization term coefficient defaults to 0.1
    Returns:
        loss - (float) the regularized loss
    """

    # add regularization term
    reg_term = 0
    if regType == 'ridge':
        reg_term = regParam*np.linalg.norm(W[1:])
    elif regType == 'lasso':
        reg_term = regParam*np.sum(np.abs(W[1:]))
    
    # compute loss
    loss = dataRDD.map(lambda x: x[1]*np.log(1 + np.exp(-np.dot(x[0], W))) + \
                             (1 - x[1])*(np.dot(x[0], W) + np.log(1 + np.exp(-np.dot(x[0], W))))).sum()\
                            /dataRDD.count()
    
    return loss


################################################################################
def GDUpdate(trainRDD, testRDD, W, nSteps = 20, regType = None, regParam=0.05, learningRate = 0.05, verbose = False):
################################################################################
    """
    Perform nSteps of regularized gradient descent step/update.
    Args:
        dataRDD - records are tuples of (y, features_array)
        W       - (array) model coefficients with bias at index 0
        regType - (str) 'ridge' or 'lasso', defaults to None
        regParam - (float) regularization term coefficient defaults to 0.1
        learningRate - (float) defaults to 0.1
    Returns:
        new_model - (array) updated coefficients, bias at index 0
        training_loss (float) training loss for new_model
        test_loss (float) test loss for new_model
    """
    # add a bias 'feature' of 1 at index 0
    augmentedTrainData = trainRDD.map(lambda x: (np.append([1.0], x[1]), x[0])).cache()
    augmentedTestData = testRDD.map(lambda x: (np.append([1.0], x[1]), x[0])).cache()
    
    # compute size of training sample
    sizeTrainSample = augmentedTrainData.count()
    
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each
    model = W
    #broadcast model
    #model = sc.broadcast(W) #uncomment this line when deploying it on the cloud
    for idx in range(nSteps):
        # add regularization term
        reg_term = np.zeros(len(model))
        if regType == 'ridge':
            reg_term = np.append(0,2*regParam*model[1:])
        elif regType == 'lasso':
            reg_term = np.append(0,regParam*np.sign(model[1:]))
    
        # compute gradient
        grad = augmentedTrainData.map(lambda x: ((1/(1 + np.exp(-np.dot(x[0], model))) - x[1])*x[0]))\
               .sum()/sizeTrainSample + reg_term
    
        #update model parameters
        new_model = model - learningRate*grad
        #new_model = sc.broadcast(new_model) #uncomment this line when deploying it on the cloud
        training_loss = LogLoss_grad(augmentedTrainData, new_model, regType=regType, regParam=regParam)
        test_loss = LogLoss_grad(augmentedTestData, new_model, regType=regType, regParam=regParam)
        
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        test_history.append(test_loss)
        model_history.append(new_model)
        
        # console output if desired
        #if verbose:
        #    print("----------")
        #    print(f"STEP: {idx+1}")
        #    print(f"training loss: {training_loss}")
        #    print(f"test loss: {test_loss}")
        #    print(f"Model: {[k for k in new_model]}")
        
        model = new_model
        #broadcast model
        #model = sc.broadcast(new_model) #uncomment this line when deploying it on the cloud
   
    return (train_history, test_history, model_history)

def makePrediction(dataRDD, W, treshProb=0.5):
    """
    Make predictions of target and compute number of: true positives, true negatives, 
    false positive, false negatives .
    Args:
        dataRDD - records are tuples of (y, features_array)
        W       - (array) model coefficients with bias at index 0
        treshProb- (float) threshold probability for imputation of positive labels
    Returns:
        pred - (rdd) predicted targets
        ntp - (integer) number of true positives
        ntn - (integer) number of true negatives
        nfp - (integer) number of false positives
        nfn - (integer) number of false negatives
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[1]), x[0])).cache()
    
    # compute prediction
    pred = augmentedData.map(lambda x: (int((1/(1 + np.exp(-np.dot(x[0], W))))>treshProb), x[1] )).cache()
    
    ntp = pred.map(lambda x: int((x[0]*x[1]) == 1)).sum()
    ntn = pred.map(lambda x: int((x[0]+x[1]) == 0)).sum()
    nfp = pred.map(lambda x: int((x[0] == 1) * (x[1] == 0))).sum()
    nfn = pred.map(lambda x: int((x[0] == 0) * (x[1] == 1))).sum()

    predlist = []

    predlist.append(ntp)
    predlist.append(ntn)
    predlist.append(nfp)
    predlist.append(nfn)

    print ("Returning", predlist)
    return (predlist)

################################################################################
#High level options
################################################################################

#This is about running Logistic Regression with various options:
# LR_OPT: Logistic Regression options
#0 - Numerical Variables only
#1 - Numerical + Categorical 
#2 - Numerical + Categorical + FFM
# NUM_OPT: Numerical Variable options
#0 - Normalized
#1 - With log
# CAT_OPT: Categorical variable options
#0 - Per-column Hash
#1 - Multi-column Hash
#2 - Hybrid Hash + Frequency
################################################################################

################################################################################
#Hyper parameters for tuning
#Number of iterations
#Learning rate
#Regularization parameter (lambda)
#Regularization type: Lasso/Ridge/Node
#Probability Threshold
################################################################################

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
        '--lr_opt',
        help='What kind of regression to run (default:0) : 0 = LR with Num, 1 = LR with Num + Cat, 2 = LR with FFM with Num + Cat',
        default=0
    )
parser.add_argument(
        '--num_opt',
        help='What kind of numerical variable transformation to use (default:0) : 0 = normalized, 1 = log',
        default=0
    )
parser.add_argument(
        '--cat_opt',
        help='What kind of categorical variable transformation to use (default:0) : 0 = per feature hash, 1 = multi-feature hash , 2 = hybrid frequency + hash',
        default=0
    )
parser.add_argument(
        '--nsteps',
        help='number of iterations to run: default at 150',
        default=150
    )
parser.add_argument(
        '--hyp_learning_rate',
        help='Learning Rate of algorithm: default at 0.1',
        default=0.1
    )
parser.add_argument(
        '--hyp_reg_type',
        help='Regularization type (default:0) : 0 = None, 1 = lasso, 2 = ridge',
        default=0
    )
parser.add_argument(
        '--hyp_reg_param',
        help='Regularization parameter: default: 0.02',
        default=0.02
    )
parser.add_argument(
        '--hyp_prob_thr',
        help='Probability threshold: default: 0.5'
    )

args = parser.parse_args()

print (args.lr_opt)
print (args.nsteps)


################################################################################
#Main loop of program
################################################################################

################################################################################
# Read in file here
################################################################################

# read in sample training data and convert to dataframe
train_sample = sc.textFile('data/sample_training.txt')\
                 .map(lambda x: x.split('\t'))\
                 .toDF().cache()

################################################################################
# Normalizing Numerical Variables
################################################################################

convert_cols = ['_1','_2','_3','_4','_5','_6','_7','_8','_9','_10','_11','_12','_13','_14']

for col in convert_cols:
    train_sample = train_sample.withColumn(col, train_sample[col].cast("double"))
train_sample = train_sample.cache()

#generate train data for homegrown solution - select only 10000 rows and only numerical features + target 
#Here instead, we should just read in the right textFile...
train_sample_red = train_sample.select(convert_cols).limit(10000).cache()

#impute missing values with averages
from pyspark.sql.functions import avg
for col in train_sample_red.columns:
    train_sample_red = train_sample_red.na.fill(round(train_sample_red.na.drop().agg(avg(col)).first()[0],1), [col])

train_sample_red_RDD = train_sample_red.rdd.map(lambda x: (x[0], np.array(x[1:]))).cache()

#normalize features
normedRDD = normalize(train_sample_red_RDD).cache()

#print(normedRDD.take(3))

#split into train, validation and test sets
train, validation, test = normedRDD.randomSplit([0.6, 0.2, 0.2], seed=42)

#compute the number of features
numb_features = np.size(normedRDD.first()[1]) 
#define baseline model, add one parameter representing the intercept
BASELINE = np.random.randn(numb_features + 1)
print (numb_features)

nSteps = int(args.nsteps)
regType = str(args.hyp_reg_type)
regParam = float(args.hyp_reg_param)
learningRate = float(args.hyp_learning_rate)

# run gradient descent
train_loss, test_loss, model = GDUpdate(train, validation, BASELINE, nSteps, regType=regType, 
                                        regParam=regParam, learningRate=learningRate, verbose = False)

##plt.plot(train_loss)
##plt.plot(test_loss)
##plt.title('Loss')
##plt.show()

#make predictions and compute metrics for treshProb = 0.5
predlist = makePrediction(validation, model[-1], 0.5)

ntp = float(predlist[0])
ntn = float(predlist[1])
nfp = float(predlist[2])
nfn = float(predlist[3])

print (ntp, ntn, nfp, nfn)

acc = (ntp+ntn)/(ntp+ntn+nfp+nfn)
prec = (ntp)/(ntp+nfp)
rec = (ntp)/(ntp+nfn)
f1 = 2*prec*rec/(prec+rec)
fpr = nfp/(ntn+nfp)
tpr = ntp/(ntp+nfn)
print('Accuracy is: ', acc)
print('Precision is: ', prec)
print('Recall is: ', rec)
print('F1 score is: ', f1)
print('False positive rate is: ', fpr)
#print('True positive rate is: ', tpr)

# [END pyspark]
