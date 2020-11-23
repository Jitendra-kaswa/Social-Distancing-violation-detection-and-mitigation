from model import Model
import cv2 
from absl import logging
import time 
import tensorflow as tf
import utils 
import numpy as np  
import os 
import time

start=time.time();

# file of video 
input_video_file = "/home/jitendra/Videos/test.mp4"
output_video_file = "converted.avi"


input_image_files = ("./img/image1.jpg","./img/image2.jpg")


def main(type = "image"):
	model = Model()
	names=utils.read_class_names("./data/classes.names")


	end=time.time()

	print(f"Execution time {end - start}")

	if type == "video" or type =="camera":
		if type == "video":
			vid = cv2.VideoCapture(input_video_file) 
		elif type =="camera":
			vid = cv2.VideoCapture(0)

		frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_per_second = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(output_video_file, codec, frame_per_second, (frame_width, frame_height))


		while True: 
			_,image = vid.read()
			if image is None:
				logging.warning("Empty Frame")
				time.sleep(0.1)
				continue


			image_size = image.shape[:2]

			image_input = tf.expand_dims(image,0)
			image_input = utils.transform_images(image_input, 416)
			predicted_bounding_box = model.predict(image_input)
			predicted_bounding_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in predicted_bounding_box]
			predicted_bounding_box = tf.concat(predicted_bounding_box, axis=0)

			final_detected_boxes ,class_names, scores=utils.box_detector(predicted_bounding_box)
			image=utils.get_human_box_detection(final_detected_boxes ,class_names, scores,names,image)
	
			if output_video_file:
				out.write(image)
			image = cv2.resize(image, (1200, 700))
			cv2.imshow('output', image)

			if cv2.waitKey(1) == ord('q'):
				break

		cv2.destroyAllWindows()

	if type=="image":
		
		for i,image_file in enumerate(input_image_files):
			image=cv2.imread(image_file)
			image_input = tf.expand_dims(image,0)

			image_input = utils.transform_images(image_input,416)	
			predicted_bounding_boxes = model.predict(image_input)

			predicted_bounding_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in predicted_bounding_boxes]
			
			predicted_bounding_box = tf.concat(predicted_bounding_box, axis=0)
			final_detected_boxes ,class_names, scores=utils.box_detector(predicted_bounding_box)
			image=utils.get_human_box_detection(final_detected_boxes ,class_names, scores,names,image)
			image = cv2.resize(image, (1200, 700))
			cv2.imwrite("output_%d.jpg"%i,image)
			cv2.imshow('output7', image)

			if cv2.waitKey(0) == ord('q'):
				cv2.destroyAllWindows()
    


if __name__ == '__main__':
	main('video')
	