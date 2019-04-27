# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 9:23
# @Author  : RIO
# @desc: 生成测试的离线模型：
import tensorflow as tf
import numpy as np
from grpc.beta import implementations
from os.path import join
from tensorflow_serving_client.protos import predict_pb2
from tensorflow_serving_client.protos import prediction_service_pb2


# make simple model
# Generate input data
x_data = np.arange(100, step=.1)
y_data = x_data + 20 * np.sin(x_data / 10)

n_samples = 1000
learning_rate = 0.01
batch_size = 100
n_steps = 500

MODEL_VERSION = '2'
OUTPUT_KEY = 'output'


# Reshape data
x_data = np.reshape(x_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

# Placeholders for batched input
x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# Do training
with tf.variable_scope('deprecate'):
	w = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
	b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0))

	y_pred = tf.matmul(x, w) + b
	loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)

	opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for _ in range(n_steps):
	indices = np.random.choice(n_samples, batch_size)
	x_batch = x_data[indices]
	y_batch = y_data[indices]
	_, loss_val = sess.run([opt, loss], feed_dict={x:x_batch, y:y_batch})

# 将训练好的模型保存在当前的文件夹下
# builder = tf.saved_model.builder.SavedModelBuilder(join("./model_name", MODEL_VERSION))
# inputs = {
# 	"x": tf.saved_model.utils.build_tensor_info(x)
# }
# output = {"y_pred": tf.saved_model.utils.build_tensor_info(y_pred)}
# prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
# 	inputs=inputs,
# 	outputs=output,
# 	method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
# )

# builder.add_meta_graph_and_variables(
# 	sess,
# 	[tf.saved_model.tag_constants.SERVING],
# 	{tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
# )
# builder.save()
# pprint("Model Saved Succeed.")
# host, port = FLAGS.server.split(':')
# channel = implementations.insecure_channel(host, int(port))
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
saver = tf.train.Saver()
model_exporter = tf.contrib.session_bundle.exporter.Exporter(saver)
model_exporter.init(sess.graph.as_graph_def(),named_graph_signatures={'inputs': tf.contrib.session_bundle.exporter.generic_signature({'x': x}),'outputs': tf.contrib.session_bundle.exporter.generic_signature({'y': y_pred})})
model_exporter.export('/Data/sladesha/test_cangjie_gateway_interface/deprecate/', tf.constant('1'), sess)
