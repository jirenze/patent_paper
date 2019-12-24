import tensorflow as tf

#定义神经网络结构相关参数
INPUT_NODE = 28
OUTPUT_NODE = 2
LAYER1_NODE = 500

#通过tf.get_variables函数获取变量。
def get_weight_variable(shape, regularizer):
	weights = tf.get_variable("weight",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
	#给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。在这里使用了add_to_collection函数将一个张量加入了一个集合
	if regularizer != None:
		tf.add_to_collection("losses",regularizer(weights))
	return weights

#定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
	#声明第一层神经网络的变量并完成前向传播过程
	with tf.variable_scope("layer1"):
		#这里通过tf.get_variable或tf.Variable没有本质区别
		weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases",[LAYER1_NODE],initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

	#类似的声明第二层神经网络的变量并完成前向传播过程
	with tf.variable_scope("layer2"):
		weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE] ,regularizer)
		biases = tf.get_variable("biases",[OUTPUT_NODE],initializer = tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases

	return layer2