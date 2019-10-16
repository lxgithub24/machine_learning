# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 15:24
# @Author  : RIO
from sklearn.datasets import load_svmlight_file


# 统计每个qid及对应的count;生成特征文件:label feature
def get_qid_count(prefix):
    qid_list = []
    feature_list = []
    with open('../data/data_file/{}.txt'.format(prefix), 'r') as f:
        lines = f.readlines()
        for line in lines:
            qid_list.append(line.split(' ')[1].split(':')[1])
            feature_str_list = line.split('#')[0].strip().split(' ')
            feature_list.append(' '.join([feature_str_list[0]] + feature_str_list[2:]))
    # 根据list生成qid:count字典
    qid_dict = {}
    for i in qid_list:
        if qid_dict.__contains__(i):
            qid_dict[i] += 1
        else:
            qid_dict[i] = 1
    # 将qid字典写文件
    with open('../data/data_file/{}_qid_count.txt'.format(prefix), 'w') as f:
        f.writelines('\n'.join([str(i) + ',' + str(j) for i, j in qid_dict.items()]))

    # 将feature写入到file
    with open('../data/data_file/{}_feature.txt'.format(prefix), 'w') as f:
        f.writelines('\n'.join(feature_list))


def get_model_data():
    # 获取压缩的x，y
    def get_x_y(prefix):
        return load_svmlight_file("../data/data_file/{}.txt".format(prefix))
    x_train, y_train = get_x_y('train')
    x_valid, y_valid = get_x_y('vali')
    x_test, y_test = get_x_y('test')

    def get_qid_count(prefix):
        qid_count_list = []
        with open("../data/data_file/{}_qid_count.txt".format(prefix), "r") as f:
            data = f.readlines()
            for line in data:
                qid_count_list.append(int(line.split(",")[1].strip()))
        return qid_count_list

    return x_train, y_train, get_qid_count('train'), x_valid, y_valid, get_qid_count('vali'), x_test, y_test, get_qid_count('test')


if __name__ == '__main__':
    # # 测试数据
    # get_qid_count('test')
    # # 训练数据
    # get_qid_count('train')
    # # 验证数据
    # get_qid_count('vali')
    print(get_model_data())
