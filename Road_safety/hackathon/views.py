from django.shortcuts import render, redirect
from .forms import LoginForm
from .models import Police, Driver
#import numpy as np
from math import sin, cos, sqrt, atan2, radians
import time

######################## ML #############

import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf
import time
#from inception_blocks_v2 import *
#from pygame import mixer
import os








detector = dlib.get_frontal_face_detector()
model = load_model('facenet_keras_weight.h5')
graph = tf.get_default_graph()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def alarm():
        mixer.init()
        mixer.music.load("/home/mukul/Desktop/machine_learning/iitg.ai/facenet/Wake-up-sounds.mp3")
        mixer.music.play(0)
        time.sleep(0.05)

        
def recognize_face(request, face_descriptor, database):
    print('encoding start.......\n')
    encoding = img_to_encoding(face_descriptor, model)
    print('encoding done.......\n')
    min_dist = 100
    identity = None
    
    #print(database.items())
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        print('entered loop in database items...\n')
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)
        if dist>6.3:
            print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            print('dist<min_dist\n')
            print('min dist = ' + str(dist))
            min_dist = dist
            identity = name
    ans = 1
    names = ['Mukul','Parag']
    for i in range( len(names) ):
        
        if identity:
            print(identity)
        else:
            print('identity is none !!')
        if int(identity) > i*40 and int(identity) <= (i+1)*40 and min_dist <=6.3:
            ans = 1
            request.session['ans']=ans
            return (names[i] ), min_dist, ans
        
    
    #if  min_dist >1.5 and min_dist <= 1.8:
            #ans=3
            #return str('Not Sure '), min_dist, ans
        
    if min_dist > 6.3:
            ans = 2
            request.session['ans']=ans
            #print(ans)
    request.session['ans']=ans
    return ('Fraud driver detected'), min_dist, ans



def extract_face_info(request, img, img_rgb, database):
    ans = -1
    faces = detector(img_rgb)
    print('Face detected')
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        print('length of faces > 0 \n')
        for face in faces:
            print('entered for loop\n')
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            if(image.size == 0):
                print('image size 0 ......\n')
                continue
            print('recognizing face................\n')
            name, min_dist, ans = recognize_face(request, image, database)
            print('face recognized............\n')
            if min_dist < 6.3:
                print('putting text !!!')
                cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, ( 255,0, 0), 2)
            else:
                cv2.putText(img,  name, (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                #alarm()
            #request.session['ans'] = ans
            #return ans
              
            
def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encoding(image, model):
        
    image = cv2.resize(image, (160, 160))
    img = image[...,::-1]
    img = np.around((img)/255.0, decimals=12)
    x_train = np.array([img])
    print('predicting!!')
    global graph
    with graph.as_default():
        embedding = model.predict(x_train)
        print('predicted')
        return embedding
    
                 
def initialize():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        #print(identity)
        database[identity] = img_path_to_encoding(file, model)
    return database


def recognize(request):
    database = initialize()
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    ans=0
    sum_all = 0
    count = 0
    #print('1')
    while True:
        #print('2')
        #graph = tf.get_default_graph()
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(img.size == 0 or img_rgb.size == 0):
            continue
                
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            print('extracting face info...\n')
            extract_face_info(request, img, img_rgb, database)
            ans=request.session['ans']
            print('ans ====================================== ' + str(ans))
            if ans==1:
                print('done !! You are safe!!')
                print(time.time()-start_time)
                count += 1
                sum_all += 1
                if time.time()-start_time > 10:
                    print('called')
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            if ans==2:
                print('Fraud driver detected !!')
                print(time.time()-start_time)
                count += 1
                sum_all += 2
                if time.time()-start_time > 10:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        if sum_all==0 and count==0:
            avg = 2
            print('avg = 2')
            #recognize(request)
        else:    
            avg = float(sum_all/count)
        print('avg = ' + str(avg))
        if avg>=1.5 and avg<=2:
            ans = 2
            print('ans set 2')
        if avg<1.5 and avg>=1:
            ans = 1
            print('ans set 1')
        print((time.time()-start_time))
        if (ans==1 or ans==2) and ((time.time()-start_time)>10):
            print('loop broken !!!!!!!!!!!!!!!!!!!\n')
            request.session['ans']=ans
            break
        request.session['ans']=ans
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#recognize(request)

















######################################
# Create your views here.

def home(request):
    return render(request, 'hackathon/home.html', {})

def login(request):
    form = LoginForm()
    if request.method=='POST':
        #print(request.POST)
        form = LoginForm(request.POST)
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password']
            for x in Police.objects.all():
                if username==x.username and password==x.password:
                    x.authent=True
                    x.save()
                    request.session['online_user_id'] = x.id
                    return redirect('hackathon:PoliceHome')
    return render(request, 'hackathon/login.html', {'form': form})
"""
def PoliceHome(request):
    online_user_id = request.session['online_user_id']
    online_user = Police.objects.get(id=online_user_id)
    #print(online_user)
    if online_user.authent:
        return render(request, 'hackathon/Police/home.html', {'online_user': online_user})
    else:
        return redirect('hackathon:login')

"""

def PoliceHome(request):
    try :
        min_dist = request.session['min_dist']
        online_user_id = request.session['online_user_id']
        online_user = Police.objects.get(id=online_user_id)
        #print(online_user)
        if online_user.authent:
            min_time = round(min_dist/40,2)
            return render(request, 'hackathon/Police/home.html', {'online_user': online_user, 'min_time': min_time})
        else:
            return redirect('hackathon:login')
    except :
        online_user_id = request.session['online_user_id']
        online_user = Police.objects.get(id=online_user_id)
        #print(online_user)
        if online_user.authent:
            return render(request, 'hackathon/Police/home.html', {'online_user': online_user})
        else:
            return redirect('hackathon:login')

    
def logout(request):
    online_user_id = request.session['online_user_id']
    online_user = Police.objects.get(id=online_user_id)
    online_user.authent = False
    online_user.save()
    return render(request, 'hackathon/logout.html', {})


def dumb(request):
    drivers = Driver.objects.all()
    return render(request, 'hackathon/Camera/dumb.html', {'drivers': drivers})

def record(request):
    ############# INITIALIZE 
    temp_min_dist=30000
    min_dist=30000
    min_dist_police_id = -1
    
    ############# GET THE DRIVER
    #print(request.POST['choice'])
    active_driver = Driver.objects.get(id=request.POST['choice'])
    
    ############# DETECTED ???????
    
    ###################
    
    recognize(request)
    ans = request.session['ans']
    if ans==1:
        active_driver.detected = False
        active_driver.save()
    if ans==2:
        active_driver.detected = True
        active_driver.save()
    
    ##################
    
    if active_driver.detected:
        print('\n\nThe driver ' + str(active_driver) + ' is a Fraud Driver !!!!!!!!\n\n')
        print('\ncar number : ' + str(active_driver.car_no))
        print('\n\nFinding nearest police station to inform ..............')
        for x in Police.objects.all():
            temp_min_dist = calcDist(active_driver.lat, active_driver.lon, x.lat, x.lon)
            #print(temp_min_dist)
            print('................')
            if temp_min_dist < min_dist:
                min_dist = temp_min_dist
                min_dist_police_id = x.id
                
                ## CORRESPONDING POLICE ?
                police = Police.objects.get(id=min_dist_police_id)
                request.session['min_dist'] = min_dist
                #print(police.username)
        
        ## FINAL POLICE
        police.detected_car.add(active_driver)
        police.save()
        #mainserver(request, police)
        
        ## SHOW DETECTED DRIVER
        print('\n\nNearest police station found ! Informed to ' + str(police.username) + '. \n\nOn duty , gonna catch : ' + str(police.detected_car.all()) + '\n\n')
        print('\n\nmin distance = ' + str(min_dist)+'\n\n')
        #recognize(request)
        return render(request, 'hackathon/Camera/record.html', {'active_driver': active_driver})
    else:
        print('\n\nDriver is not detected as a fraud driver\n\n')
    return render(request, 'hackathon/Camera/record.html', {'active_driver': active_driver})

def calcDist(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def take_driver(request):
    #print(request.POST['car_id'])
    
    ## NO MORE DETECTED
    x = Driver.objects.get(id=request.POST['car_id'])
    x.detected=False
    x.save()
    
    ## REMOVE FROM POLICE LIST
    online_user_id = request.session['online_user_id']
    online_user = Police.objects.get(id=online_user_id)
    
    online_user.detected_car.remove(x)
    online_user.save()
    
    print('\nCar number ' + str(x.car_no) + ' is caught by policeman ' + str(online_user.username) + '\n')
    return redirect('hackathon:PoliceHome')


def mainserver(request, police):
    print(police)
    return render(request, 'hackathon/Main/server.html', {'police': police})








##################### DRWOSY DRIVER DETECTION #############


## ML

from scipy.spatial import distance
import imutils
from pygame import mixer

def alarm():
        mixer.init()
        mixer.music.load("/home/mukul/Desktop/machine_learning/iitg.ai/facenet/Wake-up-sounds.mp3")
        mixer.music.play(0)
       # time.sleep(0.05)
        
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def vidCap(request):
    flag_drowsy = False
    request.session['flag_drowsy'] = flag_drowsy
    cap=cv2.VideoCapture(0)
    flag=0
    while True:
        #print('True gone !')
        ret, frame=cap.read()
        #print('frame = ' + str(frame))
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        #print(subjects)
        for subject in subjects:
            #print('for loop entered!')
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                print (flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10,325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    flag_drowsy = True
                    request.session['flag_drowsy'] = flag_drowsy
                    alarm()
                    #break
            else:
                flag = 0
        #cap.release()
        #cv2.destroyAllWindows()
        #break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    #cap.stop()

####



def drowsyHome(request):
    drivers = Driver.objects.all()
    return render(request, 'hackathon/Drowsy/drowsyHome.html', {'drivers': drivers})

def recordDrowsy(request):
    vidCap(request)
    flag_drowsy = request.session['flag_drowsy']
    print('request.method = ' + str(request.method))
    if request.method=='POST':
        try:
            print(request.POST['choice'])
            active_driver = Driver.objects.get(id=request.POST['choice'])
            request.session['active_driver_id'] = active_driver.id
        except KeyError:
            active_driver_id = request.session['active_driver_id']
            active_driver = Driver.objects.get(id=active_driver_id)
    else:
        active_driver_id = request.session['active_driver_id']
        active_driver = Driver.objects.get(id=active_driver_id)
    if flag_drowsy:
         active_driver.drowsy = True
         active_driver.save()

    if active_driver.drowsy:
        return redirect('hackathon:redirectPageToRecordDrowsy')
#        return render(request, 'hackathon/Drowsy/recordDrowsy.html', {'active_driver': active_driver})
    else:
        print('\n\n Okay \n\n')
    return render(request, 'hackathon/Drowsy/recordDrowsy.html', {})

def redirectPageToRecordDrowsy(request):
    return redirect('hackathon:recordDrowsy')
