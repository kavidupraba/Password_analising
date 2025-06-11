# Password_analising
checking_password_frequency..etc
this program will create rainbow table accroding to your passwrod list and match the passwords count
with HIBP data set to run program you need to download  HIBP dataset you can do it using 

curl -s --retry 10 --retry-all-errors --remote-name-all --parallel --parallel-max 150 "https://api.pwnedpasswords.com/range/{0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F}{0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F}{0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F}{0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F}{0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F}"

more detail about download can be found in 

https://github.com/HaveIBeenPwned/PwnedPasswordsDownloader/issues/79

after download remove the curl.log from download hash_passords

afterward Download  rockyou.txt from  or andy other word list you like if you go with rockyou.txt details are below

https://github.com/dw0rsec/rockyou.txt you can clone it usin 

git clone  https://github.com/dw0rsec/rockyou.txt.git

extract zip file using 

unzip rockyou.txt.zip  -d <your_destination>

run runing_PWEND_PASS.py 

thats it so far

Description: 
# 🛡️ Password Strength Classification at Scale  
**Using Machine Learning, Apache Spark, and Breach-Aware Hash Matching**

## 📌 Summary
This project tackles the real-world challenge of classifying password strength using machine learning — not just on toy data, but at **massive scale**. We processed over **14 million leaked passwords** using **Apache Spark**, **engineered features**, and built models like **Random Forests** and **Recurrent Neural Networks**, all while handling class imbalance and memory constraints.

## 🚀 Key Features
- ✅ 50GB+ dataset from [Have I Been Pwned](https://haveibeenpwned.com/)
- ⚡ Optimized with **Apache Spark** (PySpark, Spark MLlib)
- 🔒 Breach-aware: Passwords checked via **SHA-1 prefix-suffix join** (no tools like Hashcat used)
- 🔁 Scalable **RNN training using chunked data**
- 🔥 Custom threading with **ThreadPoolExecutor** to bypass GIL during I/O
- 🧪 Real-time memory tuning: `spark.local.dir`, partition configs, and executor memory
- 📊 Achieved **~96% accuracy**, macro F1-score of **0.68+** even under class imbalance

## 🧠 Architecture Overview

graph TD
    A[rockyou.txt + HIBP Data] -->|Preprocessing + Hashing| B[Feature Extraction]
    B --> C[Class Balancing / Downsampling]
    C --> D1[Random Forest Classifier]
    C --> D2[RNN Classifier (Chunked)]
    D1 --> E[Evaluation]
    D2 --> E
    E --> F[Web UI + Logs + Performance]


## 🗂️ Technologies Used
- Apache Spark — for distributed processing
- PySpark — Spark in Python
- pandas + Parquet — for chunked I/O
- scikit-learn — for classical ML
- PyTorch — for custom RNN
- ThreadPoolExecutor — for concurrent batch eval
- SHA-1 hashing — for breach detection via prefix/suffix
- Linux, Bash, JVM, CUDA awareness

## 📉 Results

| Model                   | Accuracy | F1 Score (Macro) | Training Time |
|-------------------------|----------|------------------|----------------|
| RNN (chunked)           | 95.81%   | 0.68             | 3903 sec       |
| Random Forest (raw)     | 99.74%   | 0.9962           | 1184 sec       |
| Random Forest (balanced)| 99.77%   | 0.9978           | 360 sec        |

> 🔍 _Strong password class detection improved dramatically after balancing and optimization._

## 🧠 Key Insights
- Spark partitions must be tuned (`file_size / 128MB`)
- Too many threads/workers = context switch hell
- Python ≠ real parallelism — JVM and CUDA do the heavy lifting
- Breach detection via prefix-suffix hash joins is efficient + scalable
- Class imbalance kills generalization — solved via downsampling

## 💥 What Makes This Unique?
Unlike most academic projects, this one:
- Handles **real-world breach data**
- Avoids tools like Hashcat, builds its own breach matcher
- Trains deep learning models without full in-memory loading
- Runs across **distributed workers using Spark**, not just on one machine

## 🤝 Credits
Built by:
- Kavindu Prabhashwara (`wijkav24@student.hh.se`)
- Waqas Ali (`waqali24@student.hh.se`)

## 📩 Want to know more?
We’re happy to talk about the architecture or share deeper insights.  
Reach out or fork the repo and try it on your own cluster!
