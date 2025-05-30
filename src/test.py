from pyspark.ml import Pipeline
from create_spark_se import stop_spark
from Ploting_and_removing_outliers import transfer3
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def turning_to_int(classified_df):
    # Convert boolean to int
    cols_to_convert = [
        "is_lower", "is_upper", "is_digit", "has_digits", 
        "has_symbols", "has_spaces", "has_special_characters", "has_letter_and_digit"
    ]
    for c in cols_to_convert:
        classified_df = classified_df.withColumn(c, F.col(c).cast("int"))
    return classified_df


def assemble_features(classified_df):
    # Vector assembler
    feature_columns = [
        "is_lower", "is_upper", "is_digit", "has_digits", 
        "has_symbols", "has_spaces", "has_special_characters", "length"
    ]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    final_df = assembler.transform(classified_df)
    return final_df


def decision_tree_classifier(final_df):
    # Decision Tree Classifier
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

    dt = DecisionTreeClassifier(labelCol="has_letter_and_digit", featuresCol="features", maxDepth=5)
    model = dt.fit(train_data)

    # Use model to predict
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="has_letter_and_digit", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print(model.toDebugString())


def main():
    classified_df, spark = transfer3(pcount=200)
    classified_df = turning_to_int(classified_df)
    final_df = assemble_features(classified_df)
    decision_tree_classifier(final_df)
    stop_spark(spark)


if __name__ == '__main__':
    main()
