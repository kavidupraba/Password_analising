import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from create_spark_se import create_spark,stop_spark
from runing_PWEND_PASS import read_parquet_file,writing_parquet_file
from Streangth_cehck_for_lage_data import setup_logger
from Strength_check_with_equal_class_split import spliting_data
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType,ArrayType
from pyspark.ml.feature import VectorAssembler,StringIndexer,IndexToString,StringIndexerModel
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

char2index={c: i+1 for i,c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:',.<>/?")}
char2index['<PAD>']=0
max_length=20

def encoding_password(pw,char2index=char2index, max_length=max_length):

    pw=(pw or "").lower()[:max_length]
    arr=[char2index.get(c,0) for c in pw]
    arr+=[0]*(max_length-len(arr))
    return arr

class Passworddataset(Dataset):

    def __init__(self,dataframe):
        super().__init__()
        self.x=dataframe["encoded_pass"].tolist()
        self.y=dataframe["label"].tolist()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return torch.tensor(self.x[idx],dtype=torch.long),torch.tensor(self.y[idx],dtype=torch.long)


class PasswordRNN(nn.Module):
    def __init__(self, vocab_size,embeded_dims,hidden_dims,num_classes=3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim=embeded_dims,padding_idx=0)
        self.rnn=nn.GRU(embeded_dims,hidden_dims,batch_first=True)
        self.fc=nn.Linear(hidden_dims,num_classes)
        #self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.embedding(x)
        _,h_n=self.rnn(x)
        out=self.fc(h_n.squeeze(0))
        return out

def main():

    logger=setup_logger(log_file='pipeline_for_RNN.log')

    logger.info("getting spark")
    spark=create_spark(partition=10,driver_memory="6g",executor_memory="10g")
    
    if not (os.path.exists("/home/jack/big_data/Project/RNN_tran.parquet") and os.path.exists("/home/jack/big_data/Project/RNN_test.parquet")):
        logger.info("reading parquet file")
        df=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/rock_you_label_v1.parquet")
        df=df.dropDuplicates()

        logger.info("printing schema")
        df.printSchema()
        # logger.info("showing_sample")
        # df.show(5,truncate=False)
        
        # logger.info("gettig fraction of the data for test run")
        # sample_df=df.sample(fraction=0.01,seed=42)
        
        logger.info("creating udf for encoding")
        encoding_udf=F.udf(encoding_password,ArrayType(IntegerType()))
        
        logger.info("apply udf to sample data set")
        encoded_df=df.withColumn("encoded_pass",encoding_udf(df["password"]))

        logger.info("droping normal password column form encoded_df")
        encoded_df=encoded_df.drop("password")
        
        train_df,test_df=spliting_data(encoded_df)


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

        writing_parquet_file(train_df,file_path="/home/jack/big_data/Project/RNN_tran.parquet")
        writing_parquet_file(test_df,file_path="/home/jack/big_data/Project/RNN_test.parquet")

    else:
        train_df=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/RNN_tran.parquet")
        test_df=read_parquet_file(spark=spark,file_path="/home/jack/big_data/Project/RNN_test.parquet")
    
    
    
    logger.info("turning data into pandas")
    #pandas_df=encoded_df.toPandas()
    pandas_df_train=train_df.toPandas()
    # pandas_df_test=test_df.limit(10000).toPandas()

    # logger.info("stoping spark")
    # stop_spark(spark=spark)
    
    logger.info("showing heade 3 lines after panda transformation")
    print(pandas_df_train.head(3))
    # print(pandas_df_test.head(3))

    #dataset=Passworddataset(pandas_df)
    dataset_train=Passworddataset(pandas_df_train)
    # dataset_test=Passworddataset(pandas_df_test)

    # loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True)
    loader_train=DataLoader(dataset=dataset_train,batch_size=32,shuffle=True)
    # loader_test=DataLoader(dataset=dataset_test,batch_size=32,shuffle=False)

    logger.info("initiating RNN model")
    model=PasswordRNN(vocab_size=len(char2index),embeded_dims=32,hidden_dims=64)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(10):
        total_loss=0
        for x_batch,y_batch in loader_train:
            optimizer.zero_grad()
            preds=model(x_batch)
            loss=criterion(preds,y_batch)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        logger.info(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")

    split_dfs=test_df.randomSplit([0.33,0.33,0.34],seed=42)
    model.eval()
    all_pred=[]
    all_labels=[]
    for i,df_part in enumerate(split_dfs):
        pd_test=df_part.toPandas()
        dataset_test=Passworddataset(pd_test)
        loader_test=DataLoader(dataset=dataset_test,batch_size=32,shuffle=False)
        with torch.no_grad():
            for x_batch,y_batch in loader_test:
                outputs=model(x_batch)
                pred=torch.argmax(outputs,dim=1)
                all_pred.extend(pred.tolist())
                all_labels.extend(y_batch.tolist())

    accuracy=accuracy_score(all_labels,all_pred)
    logger.info(f"Final Accuracy:{accuracy*100:.4f}%")

    
    # logger.info("stoping spark")
    # stop_spark(spark=spark)

    cm=confusion_matrix(all_labels,all_pred)
    logger.info(f"confusion_matrix:{cm}")

    logger.info("stoping spark")
    stop_spark(spark=spark)


    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Weak', 'Medium', 'Strong'], yticklabels=['Weak', 'Medium', 'Strong'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion_matrix")
    plt.show()

    logger.info(classification_report(all_labels, all_pred, target_names=["Weak", "Medium", "Strong"]))

    checker=input("do you want to save the model? [Y/n]")
    if checker=="Y":
        torch.save({'model_state_dict': model.state_dict(),'char2index': char2index,'params': {'embed_dim': 32, 'hidden_dim': 64}}, 'rnn_model_bundle.pt')
        logger.info("model is saved")
    else:
        pass
 


if __name__ == '__main__':
    main()

