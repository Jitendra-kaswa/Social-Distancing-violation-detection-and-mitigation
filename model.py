import tensorflow as tf 
from commonBlocks import darknet53,upsample,Convolutional_layer
import numpy as np
from load_weights import Load_weights
 

# hyperparameters 
NUMBER_OF_CLASSES = 80
STRIDES = np.array([8,16,32])
ANCHORS =(1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875)
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
weight_file = "./weights/yolov3.weights"


def yoloV3(input_layer):
	route_1_output, route_2_output, convolution_output = darknet53(input_layer) 

	convolution_output = Convolutional_layer(convolution_output, (1,1,1024,512)) 
	convolution_output = Convolutional_layer(convolution_output,(3,3,512,1024)) 
	convolution_output = Convolutional_layer(convolution_output, (1,1,1024, 512))  
	convolution_output = Convolutional_layer(convolution_output, (3,3,512,1024))  
	convolution_output = Convolutional_layer(convolution_output,(1,1,1024,512)) 

	convolution_output_lobj_branch = Convolutional_layer(convolution_output,(3,3,512,1024))
	convolution_output_lbbox = Convolutional_layer(convolution_output_lobj_branch,(1,1,1024,3*(NUMBER_OF_CLASSES+5)),
		activate= False, batch_norm = False)

	convolution_output = Convolutional_layer(convolution_output,(1,1,512,256))
	convolution_output = upsample(convolution_output)

	convolution_output = tf.concat([convolution_output, route_2_output], axis =-1) 
	convolution_output = Convolutional_layer(convolution_output,(1,1,768,256)) 
	convolution_output = Convolutional_layer(convolution_output,(3,3,256, 512))
	convolution_output = Convolutional_layer(convolution_output,(1,1,512,256))
	convolution_output = Convolutional_layer(convolution_output,(3,3,256,512))
	convolution_output = Convolutional_layer(convolution_output, (1,1,512,256))

	convolution_output_mobj_branch = Convolutional_layer(convolution_output, (3,3,256,512))
	convolution_output_mbbox = Convolutional_layer(convolution_output_mobj_branch ,(1,1,512,3*(NUMBER_OF_CLASSES+5)),
		activate= False, batch_norm= False)


	convolution_output = Convolutional_layer(convolution_output, (1,1,256,128))
	convolution_output = upsample(convolution_output)

	convolution_output = tf.concat([convolution_output,route_1_output], axis = -1)

	convolution_output = Convolutional_layer(convolution_output, (1,1,384,128))
	convolution_output = Convolutional_layer(convolution_output, (3,3,128, 256))
	convolution_output = Convolutional_layer(convolution_output, (1,1,256, 128))
	convolution_output = Convolutional_layer(convolution_output, (3,3,128, 256))
	convolution_output = Convolutional_layer(convolution_output, (1,1,256, 128))

	convolution_output_sobj_branch = Convolutional_layer(convolution_output,(3,3,128, 256))
	convolution_output_sbbox = Convolutional_layer(convolution_output_sobj_branch,
		(1,1,256,3*(NUMBER_OF_CLASSES+5)),activate= False , batch_norm= False)
	return [convolution_output_sbbox, convolution_output_mbbox, convolution_output_lbbox]


def decode(convolution_out, i = 0):
	output_convolution_shape = tf.shape(convolution_out)
	batch_size = output_convolution_shape[0]
	output_size = output_convolution_shape[1]

	convolution_output = tf.reshape(convolution_out, (batch_size, output_size,output_size, 3,5+NUMBER_OF_CLASSES))
	
	convolution_output_raw_dxdy = convolution_output[:,:,:,:,0:2]
	convolution_output_raw_dwdh = convolution_output[:,:,:,:,2:4]
	convolution_output_raw_conf = convolution_output[:,:,:,:,4:5]
	convolution_output_raw_prob = convolution_output[:,:,:,:,5:]

	y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])
	x = tf.tile(tf.range(output_size, dtype= tf.int32)[tf.newaxis,:],[output_size,1])

	xy_grid = tf.concat([x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis = -1)
	xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
	xy_grid = tf.cast(xy_grid,tf.float32)

	xy_prediction = (tf.sigmoid(convolution_output_raw_dxdy)+xy_grid)*STRIDES[i]
	wh_prediction = (tf.exp(convolution_output_raw_dwdh)*ANCHORS[i])*STRIDES[i]
	xywh_prediction = tf.concat([xy_prediction,wh_prediction], axis = -1)

	confidence_prediction = tf.sigmoid(convolution_output_raw_conf)
	probability_prediction = tf.sigmoid(convolution_output_raw_prob)

	return tf.concat([xywh_prediction, confidence_prediction, probability_prediction], axis = -1)

def Model():
	input_layer = tf.keras.layers.Input([416,416,3])
	Feature_maps = yoloV3(input_layer)

	bounding_box_tensors = []

	for i , fm in enumerate(Feature_maps):
		bounding_box_tensor = decode(fm, i)
		bounding_box_tensors.append(bounding_box_tensor)


	model = tf.keras.Model(input_layer, bounding_box_tensors)
	model = Load_weights(model, weight_file)

	return model 

#model=Model()
#model.summary()





