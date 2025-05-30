from pyspark.sql import SparkSession
from pyspark import SparkConf


def create_spark(partition=5,driver_memory="6g",executor_memory="8g"):
    conf=SparkConf()\
        .setAppName("Password Analysis")\
        .set("spark.driver.memory", driver_memory)\
        .set("spark.executor.memory", executor_memory)\
        .set("spark.sql.shuffle.partitions", str(partition))\
        .set("spark.local.dir","/home/jack/Documents/spark_space")#I have to add it because our OS not share swqp 
         #space that given to ramoverload is not sharing with
         #spark so I created another file in Documents/ called spark_space to stop it from crashing after running out of memory

    # Create a Spark session
    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()
    
    return spark

def stop_spark(spark):
    # Stop the Spark session
    spark.stop()
    

if __name__ == '__main__':
    create_spark()