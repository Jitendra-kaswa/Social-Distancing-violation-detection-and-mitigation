import tensorflow as tf 
import numpy as np 
import cv2
#%%time
from scipy.spatial import distance

def msgfunctionality():
  import clx.xms
  import requests

  client = clx.xms.Client(service_plan_id='141778dc7cf24135b74e550e5224eaef', token='0b8243d3a3174b8f8a16da06ded85562')

  create = clx.xms.api.MtBatchTextSmsCreate()
  create.sender = '+447537404817'
  create.recipients = {'+918875609015'} # this is my numberber
  create.body = 'The area is overcrowded or people are not following rules,there is high risk of infection,so please take a look at situation.'

  try:
    batch = client.create_batch(create)
  except (requests.exceptions.RequestException,clx.xms.exceptions.ApiException) as ex:
    print('Failed to communicate with XMS: %s' % str(ex))

  return 

def midpoint_point(image,person,idx):
  #get the coordinates
  X1,Y1,X2,Y2 = person[idx]
  _ = cv2.rectangle(image, (X1, Y1), (X2, Y2), (255,0,0), 1)
  
  #compute bottom center of bbox
  X_midpointpoint = int((X1+X2)/2)
  Y_midpointpoint = int(Y2)
  midpoint   = (X_midpointpoint,Y_midpointpoint)
  
  _ = cv2.circle(image, midpoint, 2, (0, 255, 0), -1)
  cv2.putText(image, str(idx), midpoint, cv2.FONT_HERSHEY_SIMPLEX,0.25, (255, 255, 255), 1, cv2.LINE_AA)
  
  return midpoint

def compute_distance(midpointpoints,number):
  dist = np.zeros((number,number))
  for i in range(number):
    for j in range(i+1,number):
      if i!=j:
        dst = distance.euclidean(midpointpoints[i], midpointpoints[j])
        dist[i][j]=dst
        dist[j][i]=dst
  return dist

def find_closest(dist,number,thresh):
  p1=[]
  p2=[]
  d=[]
  for i in range(number):
    for j in range(i,number):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
  return p1,p2,d

def change_2_red(image,person,p1,p2):
  risky = np.unique(p1+p2)
  for i in risky:
    X1,Y1,X2,Y2 = person[i]
    _ = cv2.rectangle(image, (X1, Y1), (X2, Y2), (0,0,255), 1)  
  return image

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as dt:
        for ID, name in enumerate(dt):
            names[ID] = name.strip('\n')
    return names

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def box_detector(pred):

    X_Center,Y_Center,width,height,confidence,classes = tf.split(pred,[1,1,1,1,1,-1], axis=-1)
    top_left_x=(X_Center-width/2.)/ 416
    top_left_y = (Y_Center - height / 2.0)/416.0
    bottom_right_x = (X_Center + width / 2.0)/416.0
    bottom_right_y = (Y_Center + height / 2.0)/416.0

    bxs = tf.concat([top_left_y,top_left_x,bottom_right_y,bottom_right_x],axis=-1)
    scores = confidence*classes
    scores = np.array(scores)

    scores = scores.max(axis=-1)
    class_index = np.argmax(classes, axis=-1)

    final_indxs = tf.image.non_max_suppression(bxs,scores, max_output_size= 20)
    final_indxs = np.array(final_indxs)
    class_names = class_index[final_indxs]
    bxs = np.array(bxs)
    scores = np.array(scores)
    class_names = np.array(class_names)
    bxs = bxs[final_indxs,:]

    scores = scores[final_indxs]
    bxs = bxs*416

    return bxs ,class_names, scores

def get_human_box_detection(bxs, class_names,scores,names,image):
    dt = np.concatenate([bxs,scores[:,np.newaxis],class_names[:,np.newaxis]],axis=-1)
    dt = dt[np.logical_and(dt[:, 0] >= 0, dt[:, 0] <= 416)]
    dt = dt[np.logical_and(dt[:, 1] >= 0, dt[:, 1] <= 416)]
    dt = dt[np.logical_and(dt[:, 2] >= 0, dt[:, 2] <= 416)]
    dt = dt[np.logical_and(dt[:, 3] >= 0, dt[:,3] <= 416)]
    dt = dt[dt[:,4]>0.4]
    #print(dt)

    image = cv2.resize(image, (416, 416))
    person = list()
    thresh=50
    for i,row in enumerate(dt):
        if names[row[5]]=='person':
            person.append((int(row[1]),int(row[0]),int(row[3]),int(row[2])))
    midpointpoints = [midpoint_point(image,person,i) for i in range(len(person))]
    
    number = len(midpointpoints)
    dist= compute_distance(midpointpoints,number)
    p1,p2,d=find_closest(dist,number,thresh)
    image = change_2_red(image,person,p1,p2)
    p=len(np.unique(p1+p2))
    
    ratio=(p/number)*100
    if ratio >30 :
        image = cv2.putText(image,("Social distance Violation :{}".format(p)),(10,10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0,255 ),1)
        image = cv2.putText(image,("Violation Ratio:{:.2f}".format(ratio)),(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0,255 ),1)
        # msgfunctionality()
    else:
        image = cv2.putText(image,("Social distance Violation :{}".format(p)),(10,10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
        image = cv2.putText(image,("Violation Ratio:{:.2f}".format(ratio)),(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)

    return  image