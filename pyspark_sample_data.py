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

""" Pyspark script to be uploaded to Cloud Storage and run on
Cloud Dataproc.

Note this file is not intended to be run directly, but run inside a PySpark
environment.
"""

# imports
import pyspark
import json
import time
import numpy as np

# [START pyspark]
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# read in data
start = time.time()
train_data = spark.read.format("com.databricks.spark.csv") \
    .option("delimiter", "\t") \
    .option("header", "false") \
    .load('gs://[...YOUR BUCKET NAME...]/data/train.txt')

print(f'train data read in {time.time() - start} seconds.')

# sample data
start = time.time()
train_sampdata = train_data.sample(False, 0.05, 42).cache()

print(f'train data sampled in {time.time() - start} seconds.')

#write to disk
train_sampdata.write \
	      .format("com.databricks.spark.csv") \
	      .option("delimiter", "\t") \
	      .csv('gs://[...YOUR BUCKET NAME...]/data/train_sampdata.txt')

# [END pyspark]