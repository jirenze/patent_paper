import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import paper_inference
import pandas as pd
import numpy as np

path = "2018.csv"
data = pd.read_csv(path)

label = data['是否转让']
label_ = label.values    #标签值

feature = data.iloc[:,2:]
feature_ = feature.values #特征值

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "E:\\Git\\patent_paper\\patent_paper\\"
MODEL_NAME = "model.ckpt"

DATA_FILE = "2018.csv"
NUM_FEATURES = 28

def readercsv(data_file):
	# 创建一个文件名队列对象
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(data_file),shuffle = True)
	# 创建一个TextLineReader
	reader = tf.TextLineReader(skip_header_lines = 1)
	# 使用行读取器读取csv文件内容
	key, value = reader.read(filename_queue)
	# 解码csv列并转化为tensor“张量”
	record_defaults = [[String],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
	col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28\
		= tf.decode_csv(value,record_defaults = record_defaults)
	features = tf.stack([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer)
		sess.run(tf.local_variables_initializer)
		coord = tf.train.Corrdinator()
		threads = tf.train.start_queue_runners(coord = coord)
		for i in range(2,7):
			examples = sess.run([features])
			print(examples)
		coord.request_stop()
		corrd.join(threads)



def train(patent):
	# 定义输入输出placeholder
	x_input = tf.placeholder(tf.float32,[None,paper_inference.INPUT_NODE], name="x-input")
	y_input = tf.placeholder(tf.float32,[None,paper_inference.OUTPUT_NODE], name="y-input")

	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	# 直接使用paper_inference.py定义的前向传播过程
	y = paper_inference.inference(x_input, regularizer)
	global_step = tf.Variable(0, trainable=False)

	# 定义损失函数、学习率、滑动平均操作以及训练过程
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(y_input, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, patent.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op=tf.no_op(name="train")
	
	# 初始化Tensorflow持久化类
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		# 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序完成
		for i in range(TRAINING_STEPS):
			xs, ys = patent.train.next_batch(BATCH_SIZE)
			_,loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x_input:xs, y_input:ys})
			# 每1000轮保存一次模型
			if i%1000 == 0:
				# 输出当前的训练过程。这里只输出了模型在当前训练batch上的损失函数大小。
				# 通过损失函数的大小可以大概了解训练的情况。
				# 在验证数据集上的正确率信息会有一个单独的程序来完成
				print("After {} training step(s), loss on taining batch is {}".format(step, loss_value))
				# 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
				# 比如“model.ckpt-1000” 来表示训练1000轮之后得到的模型
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main(argv = None):
	mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
	readercsv(DATA_FILE)
	# train(mnist)

if __name__ =="__main__":
	tf.app.run()