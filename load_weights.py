import numpy as np 


def Load_weights(model,weight_file):

	we_ght_file = open(weight_file, 'rb')

	major , minor, revision , seen, _ = np.fromfile(we_ght_file,dtype= np.int32, count=5)
	j=0
	for i in range(75):
		convolution_layer_name = 'conv2d_%d' %i if i>0 else 'conv2d'
		batch_normalization_layer_name = 'batch_normalization_%d' %j if j>0 else 'batch_normalization' 

		convolution_layer = model.get_layer(convolution_layer_name)
		filters = convolution_layer.filters
		kernel_size = convolution_layer.kernel_size[0]
		input_dimensions = convolution_layer.input_shape[-1]


		if i not in [58,66,74]:
			# darknet weights: [beta, gamma, mean, variance]
			batch_normalization_weights = np.fromfile(we_ght_file, dtype= np.float32, count = 4*filters)
			batch_normalization_weights = batch_normalization_weights.reshape((4,filters))[[1,0,2,3]]
			batch_normalization_layer = model.get_layer(batch_normalization_layer_name)

			j+=1

		else:
			convolution_bias = np.fromfile(we_ght_file,dtype= np.float32, count= filters)

		# darknet shape is (out_dim, input_dimensions, height,width)
		convolution_shape = (filters, input_dimensions,kernel_size,kernel_size)
		convolution_weights = np.fromfile(we_ght_file,dtype= np.float32, count= np.product(convolution_shape))

		#tf shpae (height, width, input_dimensions, out_dim)
		convolution_weights = convolution_weights.reshape(convolution_shape).transpose([2,3,1,0])


		if i not in [58,66,74]:
			convolution_layer.set_weights([convolution_weights])
			batch_normalization_layer.set_weights(batch_normalization_weights)
		else:
			convolution_layer.set_weights([convolution_weights,convolution_bias])

	assert len(we_ght_file.read(0))==0, 'failed to read all data'
	we_ght_file.close()

	return model




