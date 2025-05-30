from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from create_spark_se import stop_spark
from Ploting_and_removing_outliers import transfer3
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt



def turning_to_int(classified_df):
    # Convert boolean to int
    cols_to_convert = [
        "is_lower", "is_upper", "is_digit", "has_digits","has_upper", "has_lower",
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
    final_df.groupBy("has_letter_and_digit").count().show()
    return final_df,feature_columns


def decision_tree_classifier(final_df):
    # Decision Tree Classifier
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

    dt = DecisionTreeClassifier(labelCol="has_letter_and_digit", featuresCol="features", maxDepth=5)
    #model = dt.fit(train_data)

   
    paramGrid=ParamGridBuilder()\
        .addGrid(dt.maxDepth, [2, 4, 6])\
        .addGrid(dt.impurity, ["gini","entropy"])\
        .build()


    # Use model to predict
    #predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="has_letter_and_digit", 
        predictionCol="prediction", 
        metricName="accuracy"
    )

    # Create a CrossValidator
    crosval=CrossValidator(estimator=dt,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
    )
    # Fit the model
    cv_model = crosval.fit(train_data)
    best_model = cv_model.bestModel
    importances = best_model.featureImportances
    print("Best Model Depth:", best_model.getOrDefault("maxDepth"))
    print("Best Model Impurity:", best_model.getOrDefault("impurity"))
    print("Best Model immportance:", importances)
    print("Decision Tree Model:")
    print(best_model)



    # Make predictions
    predictions = best_model.transform(test_data)
    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)

    for metic in ["f1", "weightedPrecision", "weightedRecall"]:
        #balance between precision and recall f1 good when want to avoid false positive and false negative, 
        #weightedPrecision is average presision across all classes weighted by class frequency use when classes are imbalanced, 
        # weightedRecall Average recall acrross all classes , again weighted by class tell how well model capture positives for each class
        score =evaluator.setMetricName(metic).evaluate(predictions)
        print(f"{metic}: {score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    predictions.printSchema()
    predictions.select("password", "prediction", "has_letter_and_digit").show(10, truncate=False)
    return predictions,importances

def Confusion_matrix(predictions):
    #TP
    TP = predictions.filter((F.col("prediction") == 1) & (F.col("has_letter_and_digit") == 1)).count()
    #TN
    TN = predictions.filter((F.col("prediction") == 0) & (F.col("has_letter_and_digit") == 0)).count()
    #FP
    FP = predictions.filter((F.col("prediction") == 1) & (F.col("has_letter_and_digit") == 0)).count()
    #FN
    FN = predictions.filter((F.col("prediction") == 0) & (F.col("has_letter_and_digit") == 1)).count()

    total = TP + TN + FP + FN
    values = [TP, TN, FP, FN]
    labels = ["TP", "TN", "FP", "FN"]
    pracentage=[ (v/total)*100 for v in values]
    
    print("Confusion Matrix:")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix")
    bars=plt.bar(labels, values, color=["green", "blue", "red", "orange"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    #plt.show()

    for bar, prc in zip(bars, pracentage):
        heigt = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2,heigt+1, f"{prc:.2f}%", ha="center", va="bottom")
    plt.show()

def feature_importance(features, importances):
    print("Feature Importance:{importances}")
    # Plot feature importance
    feature_count=int(importances[0])
    im_idex=importances[1]
    importances=importances[2]

    fe=[0]*feature_count
    for i,val in enumerate(im_idex):
        fe[val]=importances[i]


    plt.figure(figsize=(10, 6))
    plt.barh(features, fe, color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.show()


def main():
    classified_df, spark = transfer3(pcount=10)
    classified_df = turning_to_int(classified_df)
    final_df,feature_column = assemble_features(classified_df)
    result,impoartnce=decision_tree_classifier(final_df)
    Confusion_matrix(result)
    feature_importance(feature_column, impoartnce)
    stop_spark(spark)
 

if __name__ == '__main__':
    main()
