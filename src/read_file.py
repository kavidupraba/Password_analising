from pyspark.sql import SparkSession
from create_spark_se import create_spark,stop_spark
from pyspark.sql.functions import col, regexp_extract, regexp_replace,split
import os

def read_file(spark, file_path):
    #file_name,file_extension = os.path.splitext(file_path)
    # Read the file into a DataFrame
    if os.path.isfile(file_path):
        #for one file like rockyou.txt
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            lines = f.readlines()
            print(f"lenth of the file is {len(lines)}")
        #Pass_rdd = spark.sparkContext.textFile("./rockyou.txt")
        Pass_rdd = spark.sparkContext.parallelize([line.strip() for line in lines])
        r=Pass_rdd.take(10)
        for i in r:
            print(i)
        return Pass_rdd
    else:
        #for directory like PWEND_PASS
        Pass_rdd=spark.read.text(file_path)\
        .withColumn("HASH",split(col("value"),":").getItem(0))\
        .withColumn("COUNT",split(col("value"),":").getItem(1).cast("int"))\
        .select("HASH","COUNT").cache()
        return Pass_rdd

def turning_to_dff(spark, rdds):
    #converting rdd to dff
    rdd_df = rdds.map(lambda x: (x,)).toDF(["password"])
    clasffied_dff = rdd_df.withColumn("is_lower",col("password").rlike("^[a-z]+$")) \
        .withColumn("is_upper",col("password").rlike("^[A-Z]+$")) \
        .withColumn("is_digit",col("password").rlike("^[0-9]+$")) \
        .withColumn("has_letter_and_digit", col("password").rlike(r"(?=.*[a-zA-Z])(?=.*\d)"))\
        .withColumn("has_digits", col("password").rlike(".*[0-9].*"))\
        .withColumn("has_upper",col("password").rlike(".*[A-Z].*")) \
        .withColumn("has_lower",col("password").rlike(".*[a-z].*")) \
        .withColumn("has_symbols",col("password").rlike("[^a-zA-Z0-9].*")) \
        .withColumn("has_spaces",col("password").rlike("\s")) \
        .withColumn("has_special_characters", col("password").rlike(r"[!@#$%^&*()_+{}\|:<>?~`\-=\[\];',./]"))
  
    #cehck=clasffied_dff[clasffied_dff["password"]=="123"].count()
    check=clasffied_dff.filter(col("password") == "123").count()
    print("123 count is ",check)
    print("Row count after before removing white space:", clasffied_dff.count())
    clasffied_dff=clasffied_dff.filter(~col("password").rlike(r"^\s*$"))#~ is bitwise not and /s is whitespace *$ is end of the line in in here we filter out the empty password
    #~col("passwort").rlike(....) tells only remove passwords that match the regex
    print("Row count after removing white space:", clasffied_dff.count())
    clasffied_dff.show(10, truncate=False)
    return clasffied_dff

def transfer(pcount=200):
    spark = create_spark(partition=pcount)
    file_path = "./rockyou.txt"
    rdd=read_file(spark, file_path)
    result=turning_to_dff(spark, rdd)
    return result,spark
    #return result

if __name__ == '__main__':
    result,spark=transfer()
    #result = transfer()
    stop_spark(spark)
