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

