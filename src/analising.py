from pyspark.sql import SparkSession
import pyspark.sql.functions as F
#from pyspark.sql.functions import col
from read_file import transfer,stop_spark


#get the password length,mosth_common , frequency of each password
def analising(clasffied_dff,spark):
    #most_common
    most_common_on_top = clasffied_dff.groupBy("password").agg(F.count("*").alias("most_common_on_top")).orderBy("most_common_on_top", ascending=False)

    #most_common_10
    most_common_on_top.show(10, truncate=False)

    #analising_according_to_length
    '''
    check if password is not empty,then tunrn it to int 0 or 1 ->20 row
    then get the length of the password->21 row
    and group by the length of the password->22 row
    and count the number of passwords with the same length->22 row
    '''
    aatl= clasffied_dff.withColumn("length",F.col("password").rlike(".+").cast("int"))\
         .withColumn("length",F.length(F.col("password")))\
         .groupBy("length").agg(F.count("*").alias("length_count")).orderBy("length", ascending=False)
    
    #password_with_most_characters
    aatl.show(10, truncate=False)

    #pattern _fequency_lower_upper_digit
    pflud= clasffied_dff.groupBy("is_lower","is_upper","is_digit").agg(F.count("*").alias("pattern_frequency")).orderBy("pattern_frequency", ascending=False)
    pflud.show(10, truncate=False)
    
    clasffied_dff.filter(F.length(F.col("password"))>250).show(1, truncate=False)

    return clasffied_dff,most_common_on_top,aatl,pflud,spark

def transfer2(pcount=200):
    classfied_dff,spark = transfer(pcount=pcount)
    clasffied_dff,most_common_on_top,aatl,pflud,spark=analising(classfied_dff,spark=spark)
    return clasffied_dff,most_common_on_top,aatl,pflud,spark

    
if __name__ == "__main__":
    classfied_dff,spark = transfer(pcount=200)
    analising(classfied_dff,spark=spark)
    stop_spark(spark)
    
    
