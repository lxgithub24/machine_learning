# 项目说明 
wget https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.rar
unrar x MQ2008.rar
mv -f MQ2008/Fold1/*.txt .

python trans_data.py train.txt mq2008.train mq2008.train.group

python trans_data.py test.txt mq2008.test mq2008.test.group

python trans_data.py vali.txt mq2008.vali mq2008.vali.group

# 文件说明：
test.txt: 原始数据

test.group: qid按照顺序从上到下，每行代表该qid出现次数

test: 每行的label（该特征对推荐的打分，1最相关，4最不相关）及其特征