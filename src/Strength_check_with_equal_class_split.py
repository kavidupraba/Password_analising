from Streangth_cehck_for_lage_data import setup_logger,creating_cm
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler,StringIndexer,IndexToString,StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from clustaring_stong_vs_weak import assemble_features
from create_spark_se import create_spark,stop_spark
from runing_PWEND_PASS import read_parquet_file,writing_parquet_file





def spliting_data(df):
    #spliting data manualy because classes are imbalance
    full_df=df.withColumn("row_id",F.monotonically_increasing_id())

    weak_df=full_df.filter(F.col("password_strength")=="weak")
    medium_df=full_df.filter(F.col("password_strength")=="medium")
    strong_df=full_df.filter(F.col("password_strength")=="Strong")

    sample_count=15000

    weak_train=weak_df.sample(False,float(sample_count)/weak_df.count(),seed=42)
    #false means we don't take same entry again and structure(withreplacement, fraction ,seed)
    medium_train=medium_df.sample(False,float(sample_count)/medium_df.count(),seed=42)
    strong_train=strong_df.sample(False,float(sample_count)/strong_df.count(),seed=42)

    train_df = weak_train.union(medium_train).union(strong_train)
    test_df=full_df.join(train_df.select("row_id"),on="row_id",how="left_anti")

    return train_df, test_df



def random_forest_spit_(df,logger,tree_count=10,number_of_folds=3,paral_train=2,model_path="/home/jack/big_data/Project/best_rf_model_v2"):

    logger.info("spliting_data_into_train_and_test")
    train_df,test_df=spliting_data(df=df)

    logger.info("loading index label model")
    index_l_model=StringIndexerModel.load("/home/jack/big_data/Project/index_label_model")
    logger.info("loading index feature model")
    index_f_model=StringIndexerModel.load("/home/jack/big_data/Project/index_feature_model")
    
    logger.info("transforming train_df to index label")
    train_df=index_l_model.transform(train_df)
    logger.info("transforming test df to index label")
    test_df=index_l_model.transform(test_df)
    
    logger.info("transforming train_df to index feature")
    train_df=index_f_model.transform(train_df)
    logger.info("transforming test_df to index feature")
    test_df=index_f_model.transform(test_df)

    feature_columns = [
        "is_lower", "is_upper", "is_digit", "has_digits", "has_upper", "has_lower",
        "has_symbols", "has_spaces", "has_special_characters", "length", "common_or_rare_index"]
        
    logger.info("*FOR train_df use assembler to assemble features (keep your features in digit foramt float or int python don't have double)")
    train_df=assemble_features(df=train_df,feature_columns=feature_columns)
    logger.info("*FOR test_df use assembler to assemble features (keep your features in digit foramt float or int python don't have double)")
    test_df=assemble_features(df=test_df,feature_columns=feature_columns)

    logger.info("starting training RANDOMFOREST")
    rf=RandomForestClassifier(labelCol="label",featuresCol="features",numTrees=tree_count)
    
    logger.info("creating parameter grid max_depth[2,4,6], impurity check both [gini,entropy]")
    #creating prametergrid
    paramgrild=ParamGridBuilder()\
    .addGrid(rf.maxDepth,[2,4,6])\
    .addGrid(rf.impurity,["gini","entropy"])\
    .build()

    #evaluter
    logger.info("creating evaluator metricName='accuracy'")
    evaluator=MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    

    #create_cross_validation
    logger.info("starging CrossValidator add parallelism  and set it to 2 like parallelism=2 if it take time")
    cv=CrossValidator(
        estimator=rf,
        evaluator=evaluator,
        estimatorParamMaps=paramgrild,
        numFolds=number_of_folds

    )

    cros_val=cv.fit(train_df)
    best_model=cros_val.bestModel
    importance=best_model.featureImportances
    logger.info("best model importance: %s",importance)
    print(f"best model feature importance:{importance}")
    print(f"best model parameteres:{best_model.extractParamMap()}")

    #making_prediction
    predictions=best_model.transform(test_df)
    accuracy=evaluator.evaluate(predictions)
    logger.info("best model accuracy:%s",accuracy)
    print(f"best model accuracy: {accuracy}")

    #doing evaluation for farther clarification
    for mat in ["f1","weightedRecall","weightedPrecision"]:
        score=evaluator.setMetricName(mat).evaluate(predictions)
        logger.info("Matirc_name:%s Score: %s",mat,score)
        print(f"Test set {mat} = {score}")

    
    logger.info("turning predcition to words")
    pred=IndexToString(inputCol="prediction",outputCol="predicted_Label",labels=index_l_model.labels)
    predictions=pred.transform(predictions)
    
    logger.info("turning real label to words")
    pred_1=IndexToString(inputCol="label",outputCol="true_Label",labels=index_l_model.labels)
    predictions=pred_1.transform(predictions)

    logger.info("turning feature back to words")
    pred_c=IndexToString(inputCol="common_or_rare_index",outputCol="common_or_rare_string",labels=index_f_model.labels)
    predictions=pred_c.transform(predictions)
    
    logger.info("saving_model")
    best_model.write().overwrite().save(model_path)

    return predictions,index_l_model


def main():
    logger=setup_logger()
    logger.info("starting mian")
    try:
        logger.info("starging spark session")
        spark=create_spark(partition=200,driver_memory="8g",executor_memory="10g")

        logger.info("reading parquet file /rock_you_label_v1.parquet")
        rock_v1_label=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/rock_you_label_v1.parquet")

        predictions,index_l_model=random_forest_spit_(df=rock_v1_label,logger=logger)

        logger.info("confusion_matrix")
        creating_cm(predictions=predictions,index_l_model=index_l_model)

        logger.info("writing result to parquet file")
        #predictions.write.parquet("result_before_truningLabelTosameSample_size.parquet")
        writing_parquet_file(df=predictions,method="overwrite",file_path="/home/jack/big_data/Project/result_after_equal_split.parquet")


    finally:
        stop_spark(spark=spark)

if __name__ == '__main__':
    main()