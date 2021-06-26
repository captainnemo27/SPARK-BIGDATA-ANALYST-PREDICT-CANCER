# BigdataAnalyst_CANCERDATA_SPARK-ML-PYTHON-HDFS
# Dao Van Thang - 18133050
# MachineLearnning by pyspark use RDD, Spark SQL, to predict cancer for a hospital.
#run hadoop

start-all.sh

# run spark
start-master.sh
start-worker.sh

#put file data to HDFS:

hadoop/bin/hdfs dfs put /thang canceryn.csv

#submit python file to pyspark : 

spark-submit --master spark://master:7077  rdfoorest.py

## see info model in txt

textRandomForestinfo.txt
