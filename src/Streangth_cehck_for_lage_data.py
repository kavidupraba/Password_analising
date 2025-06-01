from runing_PWEND_PASS import read_parquet_file,stop_spark,create_spark,writing_parquet_file
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler,StringIndexer,IndexToString,StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from clustaring_stong_vs_weak import assemble_features
import logging




def setup_logger(name=__name__,level=logging.INFO,log_file='pipeline.log'):#level set to minimum INFO,DEBUG 
    logger=logging.getLogger(name)
    if not logger.hasHandlers():
        formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')#adding format to log file
        stream_handler=logging.StreamHandler()#get handler to terminal 
        stream_handler.setFormatter(formatter)#set format
        file_handler=logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)#aff format so we can read 
        logger.setLevel(level)
        logger.addHandler(stream_handler)#go to console /terminal
        logger.addHandler(file_handler)#go to file
    return logger



def turning_string_to_index(df,label_col="password_strength",feature_col="common_or_rare"):

    #transform label to index
    index_l=StringIndexer(inputCol=label_col, outputCol="label")
    index_l_model=index_l.fit(df)
    df=index_l_model.transform(df)

    #trandform our common_or_rare
    index_f=StringIndexer(inputCol=feature_col,outputCol="common_or_rare_index")
    index_f_model=index_f.fit(df)
    df=index_f_model.transform(df)

    return df,index_l_model,index_f_model

def creating_cm(predictions,index_l_model):
    predictions_p=predictions.select("predicted_Label","true_Label","password").toPandas()
    cm=confusion_matrix(predictions_p["true_Label"],predictions_p["predicted_Label"],labels=index_l_model.labels)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=index_l_model.labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def random_forest(df,logger,tree_count=10,number_of_folds=3,paral_train=2,model_path="/home/jack/big_data/Project/best_rf_model"):

    logger.info("START TRAINING RANDOM FOREST")
    rf=RandomForestClassifier(featuresCol="features",labelCol="label",numTrees=tree_count)
    
    logger.info("SPLITING data set to train and test data")
    
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    logger.info("creating parameter grid max_depth[2,4,6], impurity check both [gini,entropy]")
    #creating prametergrid
    paramgrid=ParamGridBuilder()\
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
    crossval=CrossValidator(
        estimator=rf,
        evaluator=evaluator,
        estimatorParamMaps=paramgrid,
        numFolds=number_of_folds
    )

    cros_val=crossval.fit(train_data)
    best_model=cros_val.bestModel
    importance=best_model.featureImportances
    logger.info("best model importance: %s",importance)
    print(f"best model feature importance:{importance}")
    print(f"best model parameteres:{best_model.extractParamMap()}")

    #making_prediction
    predictions=best_model.transform(test_data)
    accuracy=evaluator.evaluate(predictions)
    logger.info("best model accuracy:%s",accuracy)
    print(f"best model accuracy: {accuracy}")

    #doing evaluation for farther clarification
    for mat in ["f1","weightedRecall","weightedPrecision"]:
        score=evaluator.setMetricName(mat).evaluate(predictions)
        logger.info("Matirc_name:%s Score: %s",mat,score)
        print(f"Test set {mat} = {score}")

    best_model.write().overwrite().save(model_path)

    return predictions






def main():
    #settingup logger
    logger=setup_logger()
    try:
        logger.info("starging spark session")
        spark=create_spark(partition=200,driver_memory="8g",executor_memory="10g")

        logger.info("reading parquet file /rock_you_label_v1.parquet")
        rock_v1_label=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/rock_you_label_v1.parquet")

        #turning_to_index
        logger.info("adding indexer to label's and common_or_rare")
        df_in,index_l_model,index_f_model=turning_string_to_index(df=rock_v1_label)
        
        logger.info("save index_l_model(index_label_model)")
        index_l_model.write().overwrite().save("/home/jack/big_data/Project/index_label_model")
        logger.info("save index_f_model(index_feature_model)")
        index_f_model.write().overwrite().save("/home/jack/big_data/Project/index_feature_model")

        feature_columns = [
        "is_lower", "is_upper", "is_digit", "has_digits", "has_upper", "has_lower",
        "has_symbols", "has_spaces", "has_special_characters", "length", "common_or_rare_index"]
        
        logger.info("use assembler to assemble features (keep your features in digit foramt float or int python don't have double)")
        df=assemble_features(df=df_in,feature_columns=feature_columns)

        logger.info("start training random forest model")
        predictions=random_forest(df=df,logger=logger)
        
        logger.info("turning predcition to words")
        pred=IndexToString(inputCol="prediction",outputCol="predicted_Label",labels=index_l_model.labels)
        predictions=pred.transform(predictions)
        
        logger.info("turning real label to words")
        pred_1=IndexToString(inputCol="label",outputCol="true_Label",labels=index_l_model.labels)
        predictions=pred_1.transform(predictions)

        logger.info("turning feature back to words")
        pred_c=IndexToString(inputCol="common_or_rare_index",outputCol="common_or_rare_string",labels=index_f_model.labels)
        predictions=pred_c.transform(predictions)

        logger.info("showing table")
        predictions.select("password","predicted_Label","true_Label","common_or_rare_string").show(10, truncate=False)

        logger.info("writing result to parquet file")
        #predictions.write.parquet("result_before_truningLabelTosameSample_size.parquet")
        writing_parquet_file(df=predictions,method="overwrite",file_path="/home/jack/big_data/Project/result_before_equal_split.parquet")



        logger.info("confusion_matrix")
        creating_cm(predictions=predictions,index_l_model=index_l_model)
        # predictions_p=predictions.select("predicted_Label","true_Label","password").toPandas()
        # cm=confusion_matrix(predictions_p["true_Label"],predictions_p["predicted_Label"],labels=index_l_model.labels)
        # disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=index_l_model.labels)
        # disp.plot(cmap="Blues")
        # plt.title("Confusion Matrix")
        # plt.show()
    
    finally:
        logger.info("finish program spark stops")
        stop_spark(spark=spark)


if __name__=='__main__':
    main()



