import os
import shutil

from pyspark.sql import SparkSession
from sklearn import datasets
import pandas as pd
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

dir = os.getcwd()



sc = SparkSession.builder.getOrCreate()

iris = datasets.load_iris()
features = iris.data
target = iris.target
data = pd.concat([features,target])
print(data)

