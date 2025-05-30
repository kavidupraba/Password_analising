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
    writing_parquet_file(df=hashed_df)
    # hashed_df.write.mode("overwrite")\
    # .parquet("rock_you_clean.parquet")

def writing_parquet_file(df,method="overwrite",file_path="./rock_you_clean.parquet"):

    match method:
        case "overwrite":
            df.write.mode("overwrite")\
            .parquet(file_path)
        case _:
            df.write.parquet(file_path)



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

    pw_df=pw_df.repartition("HASH")
    split_data=split_data.repartition("suffix")

    # match_df=split_data.join(F.broadcast(pw_df),pw_df["HASH"]==split_data["suffix"])#ithink broadcast is bad option some how program crashed after I doing this
    #data set might be small but it have huge number of entry list this might be the reason yea when I doing check I found out broadcasting is better for data set 
    # that is smaller than data_set<10MB so our rock_you.txt is 100MB +(in compress form) so this aint goon work
    match_df=split_data.join(pw_df,pw_df["HASH"]==split_data["suffix"])
    match_df.select("password","sha1","COUNT").show(10,truncate=False)
    writing_parquet_file(df=match_df, method="overwrite", file_path="/home/jack/big_data/Project/rock_you_clean_v2.parquet")


    #writing_parquet_file(df=match_df)


    stop_spark(spark=spark)#remove this when you starting working on next steps



if __name__=="__main__":
    main()
