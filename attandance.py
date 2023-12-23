import cv2
import numpy as np
import dlib
import time
import csv

def faceRegister(label_id,name,count,interval):
  '''
  label_id: id
  name: face name
  count: dateset size
  interval: time interval
  '''
  cap = cv2.VideoCapture(0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  hog_face_detector = dlib.get_frontal_face_detector()
  shape_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  
  face_descriptor_extractor = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
  
  
  start_time = time.time()
  
  collect_count = 0
  
  # csv writer
  f = open('./datas/feature.csv', 'a', newline='')
  csv_writer = csv.writer(f)
  
  while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame,(width//3,height//3))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_detections =   hog_face_detector(frame,1)

    
    # for (x,y,w,h) in faces_detections:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=5)
    
    for face in faces_detections:
      l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
      
      points = shape_detector(frame,face)
      
      for point in points.parts():
        cv2.circle(frame, (point.x, point.y), 2, (0,255,0), -1)
      
      cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
      
      if collect_count < count:
        
        now = time.time()
        
        if now - start_time > interval:
      
          face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
          
          # print(face_descriptor)
          
          face_descriptor = [f for f in face_descriptor]
          
          # print(face_descriptor)
          line = [label_id, name, face_descriptor]
          
          csv_writer.writerow(line)
          
          collect_count +=1
          
          start_time = now
          
          print(f'count:{collect_count}')
        else:
          pass      
      else:
        print('Done')
        return
        
        
    cv2.imshow('frame', frame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
  f.close()     
  cap.release()     
  cv2.destroyAllWindows()
  
def getFeatureList():
  
  label_list = []
  name_list = []
  feature_list = None
  
  with open('./datas/feature.csv', 'r') as f:
    csv_reader = csv.reader(f)
    
    for line in csv_reader:
      label_id = line[0]
      name = line[1]
      face_descriptor = eval(line[2])
      
      #print(name)
      label_list.append(label_id)
      name_list.append(name)
      
      face_descriptor = np.array(face_descriptor)
      face_descriptor = np.reshape(face_descriptor,(1,-1))
      
      if feature_list is None:
        feature_list = face_descriptor
      else:
        feature_list = np.concatenate((feature_list,face_descriptor),axis = 0)
  return label_list,name_list,feature_list
      


def faceRecognizer(threshold):
  cap = cv2.VideoCapture(0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  hog_face_detector = dlib.get_frontal_face_detector()
  shape_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  
  face_descriptor_extractor = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
  
  recog_result = {}
  
  f = open('./datas/attandance.csv', 'a', newline='')
  csv_writer = csv.writer(f)
  
  fps_time = time.time()
  
  while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame,(width//3,height//3))
  
    
    faces_detections =   hog_face_detector(frame,1)

    
    # for (x,y,w,h) in faces_detections:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=5)
    
    for face in faces_detections:
      l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
      
      points = shape_detector(frame,face)
      
      # for point in points.parts():
      #   cv2.circle(frame, (point.x, point.y), 2, (0,255,0), -1)
      
      cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
      

      face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
      
      # print(face_descriptor)
      
      
      
      label_list, name_list, feature_list = getFeatureList()
      
      
      
      face_descriptor = [f for f in face_descriptor]
      
           
      face_descriptor = np.array(face_descriptor)
      
      distances = np.linalg.norm((face_descriptor-feature_list), axis = 1)
      
      min_index = np.argmin(distances)
      
      min_distance = distances[min_index]
      
      if min_distance < threshold:
        
        
        predict_id = label_list[min_index]
        predict_name = name_list[min_index]
        
        cv2.putText(frame, f"{predict_name}: {round(min_distance,2)}",(l,b+40),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
        
        now = time.time()
        need_insert = False
        
        if predict_name in recog_result:
          if now - recog_result[predict_name] > 3:
            need_insert = True
            recog_result[predict_name] = now
          else:
            need_insert = False
        else:
          recog_result[predict_name] = now
          need_insert = True
          
        if need_insert:
          time_local = time.localtime(recog_result[predict_name])
          time_str = time.strftime('%Y-%m-%d %H:%M:%S', time_local)
          line = [predict_id, predict_name, min_distance, time_str]
          csv_writer.writerow(line)
          print(f'{time_str} write succssfully: {predict_name}')
      else:
        print('no pass')
          
    now = time.time()
    fps = 1/(now - fps_time)
    fps_time = now 
          
    cv2.putText(frame, "FPS: "+ str(round(fps,2)),(20,40),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 1)
    
    cv2.imshow('frame', frame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
  f.close()
  cap.release()     
  cv2.destroyAllWindows()

#faceRegister(1,'pk',3,3)
#faceRegister(2,'pp',3,3)
# print(feature_list)
faceRecognizer(0.5)
