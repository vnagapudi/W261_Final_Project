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
# Categorical variables
################################################################################
def getTop15FreqCols (rdd):
    
    #First get a count of rdd length
    rddLen = rdd.count()

    #Create pandas dataframe to store the top 15 values
    top15df = pd.DataFrame(columns=['col', 'top15_values', 'top15_pct_contribution'])

    for col in range(14,40):
        catRdd = rdd.map(lambda x: x.split('\t')[col]) \
                    .map(lambda x: (x,1)) \
                    .reduceByKey(lambda x,y: x + y)
    
        freqRecord = catRdd.takeOrdered(15, key=lambda x: -x[1])
        freqList = []
        freqCnt = 0
        for (k,v) in freqRecord:
            freqCnt += v
            freqList.append(str(k))
        top15df = top15df.append({'col': col, 'top15_values': freqList, 'top15_pct_contribution': 100*freqCnt/rddLen}, ignore_index=True)
    
    return(top15df)



def hashTrans (rdd):
    def createHash (elem,hlen):
        import mmh3
        hashStr = []
        for str in elem:
            hashStr.append(mmh3.hash(str) % int(hlen))
        return(hashStr)

    def create1Hot(elem, hlen):
        oneHotStr = []
        #for hashStr in elem:
        for i in range (hlen):
            if (i == elem):
                oneHotStr.append(1)
            else:
                oneHotStr.append(0)
        return(oneHotStr)

    def createCatArray(elem):
        catArray = []
        for array in elem:
            for x in array:
                catArray.append(x)
        return(np.array(catArray))

    #Define murmurHash level for 1-hot encoding
    HASHLEN = 16

    #testRdd.map(lambda x : x.split('\t')[14:40]).map(lambda x: [mmh3.hash(xn)%16 for xn in x]).take(5)
    categoricalRdd = rdd.map(lambda x : x.split('\t')[14:40]) \
                        .map(lambda x: createHash(x,HASHLEN)) \
                        .map(lambda x: [create1Hot(xn, HASHLEN) for xn in x]) \
                        .map(createCatArray)
    return(categoricalRdd)

def hybridFreqHashTrans (top15df, rdd):

    #First create Broadcast variable
    top15dfB = sc.broadcast(top15df)
    
    #set FreqThr at 90% for using frequency to bin the categories
    TOP15_FREQ_THR = 90

    #Define murmurHash level for 1-hot encoding
    HASHLEN = 16
    
    #top15df = pd.DataFrame(columns=['col', 'top15_values', 'top15_pct_contribution'])
    
    def transformRow (row, top15df, hlen):
        col = 0
        transformedRow = []
        
        for x in row:
            #Get frequency based map
            if (top15df.loc[col]['top15_pct_contribution'] > 90):
                xtrans = 15
                for (ind,topElem) in enumerate(top15df.loc[col]['top15_values']):
                    if (x == topElem):
                        xtrans = ind
            #else hash
            else:
                xtrans = mmh3.hash(x+str(col)) % int(hlen)
        
            transformedRow.append(xtrans)
            
            #Do this for all elements
            col += 1
        return (transformedRow)

    def create1Hot(elem, hlen):
        oneHotStr = []
        #for hashStr in elem:
        for i in range (hlen):
            if (i == elem):
                oneHotStr.append(1)
            else:
                oneHotStr.append(0)
        return(oneHotStr)

    def createCatArray(elem):
        catArray = []
        for array in elem:
            for x in array:
                catArray.append(x)
        return(np.array(catArray))



    categoricalRdd = rdd.map(lambda x : x.split('\t')[14:40]) \
                        .map(lambda x: transformRow(x, top15dfB.value, HASHLEN)) \
                        .map(lambda x: [create1Hot(xn, HASHLEN) for xn in x]) \
                        .map(createCatArray)
    return(categoricalRdd)

def mergeNumPlusCatRdds(elem):
    x, y = elem
    xkey , xval = x
    merge =  (xkey, np.hstack((xval,y)))
    return(merge)

################################################################################
# FFM functions
################################################################################

def fmLoss(dataRDD, w, w1,w0) :
    """
    Computes the logloss given the data and model W
    dataRDD - array of features, label
    """
    w_bc = sc.broadcast(w)
    w1_bc = sc.broadcast(w1)
    w0_bc = sc.broadcast(w0)
    def probability_value(x,W,W1,W0): 
        xa = np.array([x])
        V =  xa.dot(W)
        V_square = (xa*xa).dot(W*W)
        phi = 0.5*(V*V - V_square).sum() + xa.dot(W1.T) + W0
        return 1.0/(1.0 + np.exp(-phi))
    
    loss = dataRDD.map(lambda x: (x[0],x[1]) if x[0] == 1 else (-1, x[1])).map(lambda x:  (probability_value(x[1],w_bc.value, w1_bc.value, w0_bc.value), x[0])) \
        .map(lambda x: (1 - 1e-12, x[1]) if x[0] == 1 else ((1e-12, x[1]) if x[0] == 0  else (x[0],x[1]))) \
        .map(lambda x: -(x[1] * np.log(x[0]) + (1-x[1])*np.log(1-x[0]))).mean()
    
    
    return float(loss)

def fmGradUpdate_v1(dataRDD, w, w1, w0, alpha, regParam, regParam1, regParam0):
    """
    Computes the gradient and updates the model
    """
    
    w_bc = sc.broadcast(w)
    w1_bc = sc.broadcast(w1)
    w0_bc = sc.broadcast(w0)
    rp_bc = sc.broadcast(regParam)
    rp1_bc = sc.broadcast(regParam1)
    rp0_bc = sc.broadcast(regParam0)
    
    #Gradient for interaction term
    
    def row_grad(x, y, W, W1, W0, regParam, regParam1, regParam0):
        xa = np.array([x])
        VX =  xa.dot(W)
        VX_square = (xa*xa).dot(W*W)
        phi = 0.5*(VX*VX - VX_square).sum() + xa.dot(W1.T) + W0
        expnyt = np.exp(y*phi) 
        grad_loss = (-y/(1+expnyt))*(xa.T.dot(xa).dot(W) - np.diag(np.square(x)).dot(W))
        return 2*regParam*W + grad_loss
    
    #Gradient for Linear term
    def row_grad1(x, y, W, W1, W0, regParam, regParam1, regParam0):
        xa = np.array([x])
        VX =  xa.dot(W)
        VX_square = (xa*xa).dot(W*W)
        phi = 0.5*(VX*VX - VX_square).sum() + xa.dot(W1.T) + W0
        expnyt = np.exp(y*phi)
        grad_loss1 = (-y/(1+expnyt))*xa
        return 2*regParam1*W1 + grad_loss1
    
    #Gradient for bias term
    def row_grad0(x, y, W, W1, W0, regParam, regParam1, regParam0):
        xa = np.array([x])
        VX =  xa.dot(W)
        VX_square = (xa*xa).dot(W*W)
        phi = 0.5*(VX*VX - VX_square).sum() + xa.dot(W1.T) + W0
        expnyt = np.exp(y*phi)
        grad_loss0 = (-y/(1+expnyt))*1
        return 2*regParam0*W0 +grad_loss0
    
    batchRDD = dataRDD.sample(False, 0.01, 2019)  
    grad = batchRDD.map(lambda x: (x[0],x[1]) if x[0] == 1 else (-1, x[1])).map(lambda x: (1, row_grad(x[1], x[0], w_bc.value, w1_bc.value, w0_bc.value, rp_bc.value,rp1_bc.value,rp0_bc.value))).reduceByKey(lambda x,y: np.add(x,y))
    model = w - alpha * grad.values().collect()[0] 
    
    grad1 = batchRDD.map(lambda x: (x[0],x[1]) if x[0] == 1 else (-1, x[1])).map(lambda x: (1, row_grad1(x[1], x[0], w_bc.value, w1_bc.value, w0_bc.value, rp_bc.value,rp1_bc.value,rp0_bc.value))).reduceByKey(lambda x,y: np.add(x,y))
    model1 = w1 - alpha * grad1.values().collect()[0]
    
    grad0 = batchRDD.map(lambda x: (x[0],x[1]) if x[0] == 1 else (-1, x[1])).map(lambda x: (1, row_grad0(x[1], x[0], w_bc.value, w1_bc.value, w0_bc.value, rp_bc.value,rp1_bc.value,rp0_bc.value))).reduceByKey(lambda x,y: np.add(x,y))
    model0 = w0 - alpha * grad0.values().collect()[0]
    
    return model, model1 ,model0

def GradientDescent(trainRDD, testRDD, model, model1, model0, nSteps = 20, 
                    learningRate = 0.01, regParam = 0.01,regParam1 = 0.01,regParam0 = 0.01, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history, model1_history, model0_history = [], [], [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    model1 = wInit1
    model0 = wInit0
    for idx in range(nSteps): 
        
        ############## YOUR CODE HERE #############
        
        model, model1, model0 = fmGradUpdate_v1(trainRDD, model, model1, model0, learningRate, regParam, regParam1, regParam0)
        training_loss = fmLoss(trainRDD, model, model1, model0) 
        test_loss = fmLoss(testRDD, model, model1, model0) 
        ############## (END) YOUR CODE #############
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        test_history.append(test_loss)
        model_history.append(model)
        model1_history.append(model1)
        model0_history.append(model0)
        
        # console output if desired
        #if verbose:
        #    print("----------")
        #    print(f"STEP: {idx+1}")
        #    print(f"training loss: {training_loss}")
        #    print(f"test loss: {test_loss}")
        #    #print(f"Model: {[k for k in model]}")
   
    return train_history, test_history, model_history, model1_history, model0_history

def wInitialization(dataRDD, factor):
    nrFeat = len(dataRDD.first()[1])
    np.random.seed(int(time.time())) 
    w =  np.random.ranf((nrFeat, factor))
    w = w / np.sqrt((w*w).sum())
    
    w1 =  np.random.ranf(nrFeat)
    w1 = w1 / np.sqrt((w1*w1).sum())
    
    w0 =  np.random.ranf(1)
    
    return w, w1, w0

def fmMakePrediction(dataRDD, w, w1, w0):
    """
    Perform one regularized gradient descent step/update.
    Args:
        dataRDD - records are tuples of (y, features_array)
        W       - (array) model coefficients with bias at index 0
    Returns:
        pred - (rdd) predicted targets
    """
    w_bc = sc.broadcast(w)
    w1_bc = sc.broadcast(w1)
    w0_bc = sc.broadcast(w0)
    def predict_fm(x, W, W1, W0):
        xa = np.array([x])
        VX =  xa.dot(W)
        VX_square = (xa*xa).dot(W*W)
        phi = 0.5*(VX*VX - VX_square).sum() + xa.dot(W1.T) + W0
        return 1.0/(1.0 + np.exp(-phi))
    
    # compute prediction
    pred = dataRDD.map(lambda x: (int(predict_fm(x[1],w_bc.value, w1_bc.value, w0_bc.value)>0.5), x[0] ))
    ntp = pred.map(lambda x: int((x[0]*x[1]) == 1)).sum()
    ntn = pred.map(lambda x: int((x[0]+x[1]) == 0)).sum()
    nfp = pred.map(lambda x: int((x[0] == 1) * (x[1] == 0))).sum()
    nfn = pred.map(lambda x: int((x[0] == 0) * (x[1] == 1))).sum()
   
    #return pred, ntp, ntn, nfp, nfn
    return float(ntp), float(ntn), float(nfp), float(nfn)
   
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

#Read in the complete dataset
train_sample_rdd = sc.textFile('data/sample_training.txt')

#Get the top 1000 rows only (as before for numerical variables)
testRdd = sc.parallelize(train_sample_rdd.take(10000),1)

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

#split into train, validation and test sets
if (args.lr_opt == 0):
    train, validation, test = normedRDD.randomSplit([0.6, 0.2, 0.2], seed=42)
    numb_features = np.size(normedRDD.first()[1]) 
elif (args.lr_opt == 1):
    if (args.cat_opt == 0):
        categoricalRDD = hashTrans(testRdd)
    elif (args.cat_opt == 1):
        top15df = getTop15FreqCols(train_sample_rdd)
        categoricalRDD = hybridFreqHashTrans(top15df, testRdd)
        numPlusCatRdd = normedRDD.zip(categoricalRdd) \
                         .map(mergeNumPlusCatRdds)
    train, validation, test = numPlusCatRDD.randomSplit([0.6, 0.2, 0.2], seed=42)
    numb_features = np.size(numPlusCatRDD.first()[1]) 

#define baseline model, add one parameter representing the intercept
BASELINE = np.random.randn(numb_features + 1)
print (numb_features)

nSteps = int(args.nsteps)
regType = str(args.hyp_reg_type)
regParam = float(args.hyp_reg_param)
learningRate = float(args.hyp_learning_rate)


if ((args.lr_opt == 0) or (args.lr_opt == 1)) :
    # run gradient descent
    train_loss, test_loss, model = GDUpdate(train, validation, BASELINE, nSteps, regType=regType, 
                                        regParam=regParam, learningRate=learningRate, verbose = False)
else:
    wInit, wInit1, wInit0 = wInitialization(train, 2)
    logerr_train, logerr_test, models, model1s, model0s = GradientDescent(train, validation, wInit, wInit1, wInit0, nSteps = 100,
                                                    learningRate = 0.002, regParam = 0.01, regParam1 = 0.01, regParam0 = 0.01, verbose = True)

##plt.plot(train_loss)
##plt.plot(test_loss)
##plt.title('Loss')
##plt.show()

#make predictions and compute metrics for treshProb = 0.5
if (args.lr_opt == 0) or (args.lr_opt == 1) :
    predlist = makePrediction(validation, model[-1], 0.5)
else:
    ntp, ntn, nfp, nfn = fmMakePrediction(validation, models[-1], model1s[-1], model0s[-1])

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
