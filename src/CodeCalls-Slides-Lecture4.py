
from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession.builder.appName("FlightData").getOrCreate()
flightData2015 = spark.read.option("inferSchema", "true").option("header", "true").csv("mydata.csv")

result=flightData2015.take(3)
print(f"HERE IS THE RESULT: {result} it's type is {type(result)}")


flightData2015.sort("count").explain()


spark.conf.set("spark.sql.shuffle.partitions", "5")

flightData2015.sort("count").take(5)

flightData2015.createOrReplaceTempView("flight_data_2015")

sqlWay = spark.sql("SELECT DEST_COUNTRY_NAME, count(1) FROM flight_data_2015 GROUP BY DEST_COUNTRY_NAME")

dataFrameWay = flightData2015.groupBy("DEST_COUNTRY_NAME").count()

from pyspark.sql.functions import max
flightData2015.select(max("count")).take(1)#we import sql max function then use it to get the maximum value from the count column

maxSql = spark.sql("SELECT DEST_COUNTRY_NAME, sum(count) as destination_total FROM flight_data_2015 GROUP BY DEST_COUNTRY_NAME ORDER BY sum(count) DESC LIMIT 5")
dataFrameWay=flightData2015.groupBy("DEST_COUNTRY_NAME").sum("count").alias("destination_total")
dataFrameWay = dataFrameWay.orderBy(dataFrameWay["sum(count)"].desc()).limit(5)
#dataFrameWay = dataFrameWay.sort(dataFrameWay["sum(count)"].desc()).limit(5)

dataFrameWay.show()

maxSql.show()

spark.stop()