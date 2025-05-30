from pyspark.sql import SparkSession,DataFrame
from create_spark_se import create_spark,stop_spark
import pyspark.sql.functions as F
from read_file import read_file
from Ploting_and_removing_outliers import transfer3
import os



def creating_parquet_file(pcount=10):
    #get clean rockyou.txt file (2009)
    clean_df,spark=transfer3(pcount=pcount)
    password_only=clean_df.select("password")
    password_only.write.parquet("rock_you_clean.parquet")
    stop_spark(spark=spark)

def read_parquet_file(spark,file_path="./rock_you_clean.parquet"):
    df=spark.read.parquet(file_path).repartition(10)
    df.show(5,truncate=False)
    return df

def turning_to_hash(df):
    hashed_df=df.withColumn("sha1",F.upper(F.sha1(F.col("password"))))
    hashed_df.write.mode("overwrite")\
    .parquet("rock_you_clean.parquet")


def main():

    if not os.path.exists("./rock_you_clean.parquet"):
        creating_parquet_file(pcount=10)


    spark=create_spark(partition=200,driver_memory="8g",executor_memory="10g")
    path="./PWEND_PASS"
    pw_df=read_file(spark=spark,file_path=path)

    
    #read_parquet_file(spark=spark)
   

    # if isinstance(rdd,DataFrame):
    #     rdd=rdd.rdd

    # rdd_df=rdd.map(lambda pwd: (pwd,)).toDF(["HASH_PASS"]).cache()#cashe will store our data in ram and remain in our swap partion this will greatly 
    # reduce process 
    # #time after my program session finishes or I run spark.stop() it will automaticaly release the cache if you crash kill same thing happens
    # we can check this before and after runing htop or freeh 
    pw_df.show(10, truncate=False)
    #length_check=rdd.withColumn("length",F.length(F.col("HASH"))==40)
    #length_check.show(10,truncate=False)
    #length_check.filter(F.col("length")==True).show(1,truncate=False)
    before_sha1=read_parquet_file(spark=spark)

    #creating hash log
    turning_to_hash(before_sha1)

    after_sha1=read_parquet_file(spark=spark)

    split_data=after_sha1.withColumn("prefix",F.expr("substring(sha1,1,5)"))\
    .withColumn("suffix",F.expr("substring(sha1,6,35)"))

    pw_df=pw_df.repartition(200,"HASH")
    split_data=split_data.repartition(200,"suffix")

    match_df=split_data.join(F.broadcast(pw_df),pw_df["HASH"]==split_data["suffix"])
    match_df.select("password","sha1","COUNT").show(10,truncate=False)


    stop_spark(spark=spark)#remove this when you starting working on next steps



if __name__=="__main__":
    main()
