
echo "Winogrande"
cat $1/winogrande.txt | grep 'test:1267/1267 | accuracy'
echo "PiQA"
cat $1/piqa.txt | grep 'test:1838/1838 | accuracy'
echo "SCIQ"
cat $1/social_i_qa.txt | grep 'test:1954/1954 | accuracy'
echo "OBQA"
cat $1/openbookqa.txt | grep 'test:500/500 | accuracy'
echo "HS"
cat $1/hellaswag.txt | grep 'test:10042/10042 | accuracy'
echo "BOOLQ"
cat $1/boolq.txt | grep 'test:3270/3270 | accuracy'
echo "ARC-E"
cat $1/ARC-Easy.txt | grep 'test:2376/2376 | accuracy'
echo "ARC-C"
cat $1/ARC-Challenge.txt | grep 'test:1172/1172 | accuracy'


