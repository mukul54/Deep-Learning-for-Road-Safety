from django.shortcuts import render, redirect
from .forms import LoginForm
from .models import Police, Driver
#import numpy as np
from math import sin, cos, sqrt, atan2, radians
import time

######################## ML #############
"""
import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import time
#from inception_blocks_v2 import *
from pygame import mixer
import os


detector = dlib.get_frontal_face_detector()
model = load_model('facenet_keras_weight.h5')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def alarm():
        mixer.init()
        mixer.music.load("/home/mukul/Desktop/machine_learning/iitg.ai/facenet/Wake-up-sounds.mp3")
        mixer.music.play(0)
        time.sleep(0.05)

        
def recognize_face(face_descriptor, database):
    encoding = img_to_encoding(face_descriptor, model)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    ans = 1
    names = ['Mukul','Jayant','Chandan']
    for i in range( len(names) ):
        if int(identity) > i*10 and int(identity) <= (i+1)*10 and min_dist <=1.5:
            ans = 1
            return (names[i] + " " + str(ans) ), min_dist
        
    
    if  min_dist >1.5 and min_dist <= 1.8:
            ans=3
            return str('Not Sure ' + str(ans) ), min_dist
        
    if min_dist > 1.8:
            ans = 2
            print(ans)
            return ('Not a member ' + str(ans)), min_dist



def extract_face_info(img, img_rgb, database):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            if(image.size == 0):
                continue
            name, min_dist = recognize_face(image, database)
            
            if min_dist < 1.8:
                cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, ( 255,0, 0), 2)
            else:
                cv2.putText(img,  name, (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                alarm()
              
            
def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encoding(image, model):
        
    image = cv2.resize(image, (160, 160))
    img = image[...,::-1]
    img = np.around((img)/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict(x_train)
    return embedding
    
                 
def initialize():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, model)
    return database


def recognize():
    database = initialize()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(img.size == 0 or img_rgb.size == 0):
            continue
                
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            extract_face_info(img, img_rgb, database)
        
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#recognize()














"""


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

def PoliceHome(request):
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
                #print(police.username)
        
        ## FINAL POLICE
        police.detected_car.add(active_driver)
        police.save()
        #mainserver(request, police)
        
        ## SHOW DETECTED DRIVER
        print('\n\nNearest police station found ! Informed to ' + str(police.username) + '. \n\nOn duty , gonna catch : ' + str(police.detected_car.all()) + '\n\n')
        print('\n\nmin distance = ' + str(min_dist)+'\n\n')
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
