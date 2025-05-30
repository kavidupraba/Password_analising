import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler,StringIndexer,IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from create_spark_se import stop_spark
from Ploting_and_removing_outliers import transfer3
from classification import turning_to_int,feature_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def defining_srong_weak(df):
    # Define strong and weak passwords
    df =df.withColumn("score", F.col("has_upper")+F.col("has_lower")+F.col("has_digits")+ F.col("has_special_characters")+ F.when(F.col("length")>8,1).otherwise(0))

    df = df.withColumn("Pasword_streangth",F.when(F.col("score")==5 ,"Strong").when(F.col("score")>=3,"medium").otherwise("weak"))
    
    return df

def assemble_features(df):
    # Vector assembler
    feature_columns = [
        "is_lower", "is_upper", "is_digit", "has_digits","has_upper", "has_lower",
        "has_symbols", "has_spaces", "has_special_characters", "length"
    ]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    final_df = assembler.transform(df)
    final_df.groupBy("Pasword_streangth").count().show()
    return final_df,feature_columns


def random_forest_classifier(final_df):
    index=StringIndexer(inputCol="Pasword_streangth",outputCol="label")
    index_moel=index.fit(final_df)
    final_df=index_moel.transform(final_df)
    #final_df=index.fit(final_df).transform(final_df)
    # Random Forest Classifier
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    #model = rf.fit(train_data)

    paramGrid=ParamGridBuilder()\
        .addGrid(rf.maxDepth, [2, 4, 6])\
        .addGrid(rf.impurity, ["gini","entropy"])\
        .build()

    # Use model to predict
    #predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )

    # Create a CrossValidator
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)  # Use 3+ folds in practice
    # Fit the model
    cv_model = crossval.fit(train_data)
    best_model = cv_model.bestModel
    importance=best_model.featureImportances
    print(f"best model feature importance: {importance}")
    print(f"Best Model Parameters: {best_model.extractParamMap()}")
    
    # Make predictions
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print(f"Test set accuracy = {accuracy}")
    
    for mat in ["f1","weightedRecall","weightedPrecision"]:
        score=evaluator.setMetricName(mat).evaluate(predictions)
        print(f"Test set {mat} = {score}")

    pred=IndexToString(inputCol="prediction",outputCol="predicted_label",labels=index_moel.labels)
    predictions=pred.transform(predictions)
    pred_1=IndexToString(inputCol="label", outputCol="true_label", labels=index_moel.labels)
    predictions=pred_1.transform(predictions)
    lables=index_moel.labels
    return predictions, accuracy, importance,lables



def plots(predictions, labels):
    print("schema of the predcitions ")
    print(predictions.printSchema())
    pd_pred = predictions.select("password", "predicted_label", "true_label").toPandas()

    sns.countplot(data=pd_pred, x="predicted_label", order=labels)
    plt.title("Predicted Password Strength Distribution")
    plt.xlabel("Password Strength")
    plt.ylabel("Count")
    plt.show()

    cm = confusion_matrix(pd_pred["true_label"], pd_pred["predicted_label"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


    





def main():
    # Load data
    clean_cldf,spark=transfer3(pcount=10)
    #turning to int
    int_df=turning_to_int(clean_cldf)
    # Define strong and weak passwords
    df = defining_srong_weak(int_df)
    # Assemble features
    final_df, feature_columns = assemble_features(df)
    # Convert labels to integers
    final_df = turning_to_int(final_df)
    # Train Random Forest Classifier
    predictions, accuracy , importance ,labels= random_forest_classifier(final_df)
    predictions.select("password","Pasword_streangth", "label", "prediction", "predicted_label").show(10)
    plots(predictions,labels)
    feature_importance(feature_columns,importance)

   

    
    # Stop Spark session
    stop_spark(spark)


if __name__ == "__main__":
    main()


'''
schema of the predcitions 
root
 |-- password: string (nullable = true)
 |-- is_lower: integer (nullable = true)
 |-- is_upper: integer (nullable = true)
 |-- is_digit: integer (nullable = true)
 |-- has_letter_and_digit: integer (nullable = true)
 |-- has_digits: integer (nullable = true)
 |-- has_upper: integer (nullable = true)
 |-- has_lower: integer (nullable = true)
 |-- has_symbols: integer (nullable = true)
 |-- has_spaces: integer (nullable = true)
 |-- has_special_characters: integer (nullable = true)
 |-- length: integer (nullable = true)
 |-- score: integer (nullable = true)
 |-- Pasword_streangth: string (nullable = false)
 |-- features: vector (nullable = true)
 |-- label: double (nullable = false)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)
 |-- predicted_label: string (nullable = true)
 |-- ture_label: string (nullable = true)

'''