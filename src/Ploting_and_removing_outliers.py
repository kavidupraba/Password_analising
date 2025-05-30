from pyspark.sql import SparkSession
from analising import transfer2,stop_spark
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Plotting_out(clasffied_dff=None):
    #clasffied_dff,most_common_on_top,aatl,pflud,spark=transfer2(pcount=200)
    lenghth_dff=clasffied_dff.select("password").withColumn("length",F.length(F.col("password"))).sample(False, 0.01,seed=42)\
         .limit(10000).toPandas()

    #boxplot
    sns.boxenplot(x=lenghth_dff["length"])
    plt.title("Boxplot of password length")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()
    #return spark

def removeing_outliers(clasiffied_dff):
    clasiffied_dff=clasiffied_dff.withColumn("length",F.length(F.col("password")))
    clean_cldf=clasiffied_dff.filter(F.col("length")<20)

    print("dataframe after removing outliers...10 sample")
    clean_cldf.show(10, truncate=False)
    return clean_cldf
#@overide  
def transfer3(pcount):
    clasffied_dff,most_common_on_top,aatl,pflud,spark=transfer2(pcount=pcount)
    Plotting_out(clasffied_dff)
    clean_cldf=removeing_outliers(clasffied_dff)
    return clean_cldf,spark

if __name__ == "__main__":
    clean_cldf,spark=transfer3(pcount=200)
    #spark.stop()
    stop_spark(spark)
    #stop_spark(spark)
