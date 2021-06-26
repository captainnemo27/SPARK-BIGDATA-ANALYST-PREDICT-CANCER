from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest,RandomForestModel
from pyspark.mllib.util import MLUtils
from time import *
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pykitml as pk

spark = SparkSession.builder.appName('cancerMachineLearning.com').getOrCreate()

RANDOM_SEED = 13579
TRAINING_DATA_RATIO = 0.7
RF_NUM_TREES = 1500
RF_MAX_DEPTH = 3
RF_NUM_BINS = 14
RF_MAX_BINS = 15


df = spark.read.option("header",True) \
        .csv("hdfs://master:8082/thangdao/canceryn.csv")
df.printSchema()
df.select("*").show()
df = df.replace("Yes","1")
df = df.replace('No','0')

# The last column contains the classification outcome. Turn this into an RDD
# of Labeled Points.
transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

# Split the data into a training set and a test set.
splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
training_data, test_data = transformed_df.randomSplit(splits, RANDOM_SEED)
print("Number of training set rows: %d" % training_data.count())
print("Number of test set rows: %d" % test_data.count())


start_time = time()

# Let's make sure we have the correct types.
print("%s should be an RDD" % type(training_data))
print("%s should be a LabeledPoint" % type(training_data.first()))

# Train our random forest model.
model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={}, \
    numTrees=RF_NUM_TREES, featureSubsetStrategy="auto", impurity="gini", \
    maxDepth=RF_MAX_DEPTH, maxBins=RF_MAX_BINS, seed=RANDOM_SEED)

#print("Learned classification forest model:")
#print(model.toDebugString())

end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)



predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
model_accuracy = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (model_accuracy * 100))


start_time = time()

metrics = BinaryClassificationMetrics(labels_and_predictions)
print("Area under Precision/Recall (PR) curve: %.f" % (metrics.areaUnderPR * 100))
print("Area under Receiver Operating Characteristic (ROC) curve: %.3f" % (metrics.areaUnderROC * 100))

end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate model: %.3f seconds" % elapsed_time)



output = labels_and_predictions .toDF(["du_lieu_goc","du_doan"])
output.show()
DULIEU = [(1),(2),(1),(4),(5),(6),(1),(5),(9)]
DULIEU = spark.sparkContext.parallelize(DULIEU)
start_time1 = time()
end_time1 = time()
elapsed_time1 = end_time1 - start_time1


getDuDoan = model.predict(DULIEU)


#output txt
df.write.csv(path='hdfs://master:8082/thangdao/cancer',header=True,sep=',')
output.write.csv(path='hdfs://master:8082/thangdao/cancer_output',header=True,sep=',')


file2write=open("textRandomForestinfo.txt",'w')
file2write.write("Number of training set rows: %d" % training_data.count() + "\n"
                + "Number of test set rows: %d" % test_data.count() + "\n"
                +"Model accuracy: %.3f%%" % (model_accuracy * 100)+ "\n"
                + "Time to train model: %.3f seconds" % elapsed_time + "\n"
                + "Area under Precision/Recall (PR) curve: %.f" % (metrics.areaUnderPR * 100) +"\n"
                + "Area under Receiver Operating Characteristic (ROC) curve: %.3f" % (metrics.areaUnderROC * 100) + "\n"
                + "Time to evaluate model: %.3f seconds" % elapsed_time + "\n")
file2write.close()
# collect the RDD to a list
listDD = getDuDoan.collect()
 
  # print the list
for element in listDD:
    print(element)
