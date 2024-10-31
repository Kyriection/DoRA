nohup bash script/1b/APOLLO_mini.sh 0 apollo_mini_64_1e-4 1e-4 64 > log_apollo_mini_64_1e-4.txt 2>&1 &
nohup bash script/1b/APOLLO_mini.sh 1 apollo_mini_128_3e-4 3e-4 128 > log_apollo_mini_128_3e-4.txt 2>&1 &



nohup bash script/1b/APOLLO_SVD.sh 2 apollo_svd 5e-5 > log_apollo_svd_5e-5.txt 2>&1 &
nohup bash script/1b/APOLLO_SVD.sh 3 apollo_svd 5e-4 > log_apollo_svd_5e-4.txt 2>&1 &




nohup bash script/1b/APOLLO_mini.sh 0 3e-4 64 > log_apollo_mini_64_3e-4.txt 2>&1 &
nohup bash script/1b/APOLLO_mini.sh 1 2e-4 128 > log_apollo_mini_128_2e-4.txt 2>&1 &
nohup bash script/1b/APOLLO_mini.sh 2 5e-4 128 > log_apollo_mini_128_5e-4.txt 2>&1 &
nohup bash script/1b/APOLLO_mini.sh 3 1e-4 128  > log_apollo_mini_128_1e-4.txt 2>&1 &
