import tensorflow as tf 

class BatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, x, training= False):
		if not training:
			training = tf.constant(False)
		training = tf.logical_and(training, self.trainable)
		return super().call(x, training)




def Convolutional_layer(input_layer, filters_shape, down_sample = False,
		activate = True, batch_norm = True, regularization = 0.0005, reg_stddev = 0.01, activate_alpha = 0.1):

	if down_sample:
		input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
		padding ="valid"
		strides = 2
	else:
		padding ="same"
		strides = 1
	conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
		kernel_size = filters_shape[0],
		strides = strides,
		padding = padding,
		use_bias = not batch_norm,
		kernel_regularizer= tf.keras.regularizers.l2(regularization),
		kernel_initializer = tf.random_normal_initializer(stddev=reg_stddev),
		bias_initializer = tf.constant_initializer(0.)
		)(input_layer)

	if batch_norm:
		conv = BatchNormalization()(conv)
	if activate:
		conv = tf.nn.leaky_relu(conv, alpha= activate_alpha)

	return conv

def residual_black(input_layer, input_channel, filter_num1, filter_num2):
	short_cut = input_layer
	conv = Convolutional_layer(input_layer, filters_shape=(1,1,input_layer,filter_num1))
	conv = Convolutional_layer(conv, filters_shape=(3,3,filter_num1,filter_num2))

	res_output = short_cut+ conv 
	return res_output

def darknet53(data_input):
	data_input = Convolutional_layer(data_input,(3,3,3,32))
	data_input = Convolutional_layer(data_input, (3,3,32,64), down_sample = True)

	for i in range(1):
		data_input = residual_black(data_input, 64,32,64)

	data_input = Convolutional_layer(data_input, (3,3,64,128),down_sample=True)

	for i in range(2):
		data_input = residual_black(data_input, 128,64,128)

	data_input = Convolutional_layer(data_input, (3,3,128,256), down_sample= True)

	for i in range(8):
		data_input = residual_black(data_input,256,128,256)


	output_1 = data_input 

	data_input = Convolutional_layer(data_input,(3,3,256,512), down_sample= True)

	for i in range(8):
		data_input = residual_black(data_input,512,256,512)
	output_2 = data_input
	data_input = Convolutional_layer(data_input,(3,3,512,1024), down_sample= True)

	for i in range(4):
		data_input= residual_black(data_input,1024,512,1024)


	return output_1, output_2, data_input

def upsample(input_layer):
	return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),
		method='nearest')

