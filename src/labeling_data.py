import pyspark.sql.functions as F
#from pyspark.sql import SparkSession
from clustaring_stong_vs_weak import defining_srong_weak,turning_to_int
from runing_PWEND_PASS import read_parquet_file,stop_spark,create_spark,writing_parquet_file


def creating_featuers(df):
     feature_dff = df.withColumn("is_lower",F.col("password").rlike("^[a-z]+$")) \
        .withColumn("is_upper",F.col("password").rlike("^[A-Z]+$")) \
        .withColumn("is_digit",F.col("password").rlike("^[0-9]+$")) \
        .withColumn("has_letter_and_digit", F.col("password").rlike(r"(?=.*[a-zA-Z])(?=.*\d)"))\
        .withColumn("has_digits", F.col("password").rlike(".*[0-9].*"))\
        .withColumn("has_upper",F.col("password").rlike(".*[A-Z].*")) \
        .withColumn("has_lower",F.col("password").rlike(".*[a-z].*")) \
        .withColumn("has_symbols",F.col("password").rlike("[^a-zA-Z0-9].*")) \
        .withColumn("has_spaces",F.col("password").rlike("\s")) \
        .withColumn("has_special_characters", F.col("password").rlike(r"[!@#$%^&*()_+{}\|:<>?~`\-=\[\];',./]"))\
        .withColumn("length",F.length(F.col("password")))\
        .withColumn("common_or_rare",F.when(F.col("COUNT")>=10,"common").otherwise("rare"))
     return feature_dff

def main():
    spark=create_spark(partition=200,driver_memory="8g",executor_memory="10g")
    rock_v2=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/rock_you_clean_v2.parquet")
    feature_df=creating_featuers(rock_v2)
    feature_in_df=turning_to_int(feature_df)
    feature_df.show(10,truncate=False)
    label_df=defining_srong_weak(df=feature_in_df)
    label_df.show(10,truncate=False)
    writing_parquet_file(df=label_df,file_path="/home/jack/big_data/Project/rock_you_label_v1.parquet")
    stop_spark(spark=spark)


if __name__ == "__main__":
    main()
