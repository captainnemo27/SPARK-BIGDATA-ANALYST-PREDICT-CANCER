# BigdataAnalyst_CANCERDATA_SPARK-ML-PYTHON-HDFS
# Dao Van Thang - 18133050
Use MachineLearnning in Spark ,RDD, Spark SQL, to predict cancer for a hospital.
* Version <table>

    <tr>
        <td>Hadoop</td>
        <td>2.2.7</td>
    </tr>
    <tr>
        <td>Java</td>
        <td>1.8</td>
    </tr>
    <tr>
        <td>Spark</td>
        <td>3.0.3</td>
    </tr>
   </table>
# *BUILD PROJECT*
* run hadoop:

```start-all.sh```

* run spark:
```start-master.sh```
```start-worker.sh```

* put file data to HDFS:

```hadoop/bin/hdfs dfs put /thang canceryn.csv```

* submit python file to pyspark : 

```spark-submit --master spark://master:7077  rdfoorest.py```

## see info model in txt

textRandomForestinfo.txt
