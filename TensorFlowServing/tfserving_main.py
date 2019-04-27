# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 11:16
# @Author  : RIO
# @desc: 生成调起文件：（需要自行生成/Data/muc/serving/tensorflow_serving/test文件夹）
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow_serving_client.protos import predict_pb2
from tensorflow_serving_client.protos import prediction_service_pb2
from tensorflow.python.framework import dtypes

flags.DEFINE_string('server', '127.0.0.1:9005', 'PredictionService host:port')
FLAGS = flags.FLAGS

n_samples = 100

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Generate deprecate data
x_data = np.arange(n_samples, step=1, dtype=np.float32)
x_data = np.reshape(x_data, (n_samples, 1))

# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'deprecate'
request.inputs['x'].ParseFromString(tf.contrib.util.make_tensor_proto(x_data, dtype=dtypes.float32, shape=[100, 1]).SerializeToString())
result = stub.Predict(request, 10.0) # 10 secs timeout