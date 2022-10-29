'''
 TCP  - server to do face recognetion
'''
import socket
import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2 
from time import sleep

HOST ='localhost'                                       # Server IP
PORT = 6282                                             # Server Port
BUF_SIZE = 1024                                         # Buffer Size

path = 'Images'                                         #images file path
images = []
classNames = []                                         #list contain all the images in the file
myList = os.listdir(path)

for cl in myList:                                      #loop to extract the name of the emplyee from his Image
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#------------------------------------------------------------------#
#Functions
#------------------------------------------------------------------#
def findEncodings(images):                              # function to encode the saved Images 
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList 

def markAttendance(name):                               # function to write the attendance in csv file
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n {name},{dtString}')



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # initialize a socket
s.bind((HOST,PORT))                                     # register port with OS
print("I am waiting for a client ...")
s.listen(1)                                             # wait for a connection

conn, add = s.accept()                                  # accept connection and retrieve conn object and address
print("I got a connection form ",add)
encodeListKnown = findEncodings(images)
cap = cv2.VideoCapture(0)                               #initialize the webcam
data = conn.recv(BUF_SIZE).decode()                 # check if data accepted
while True:                                             #while loop to get each frame from the camera & send the result to the client
    if str(data) !='Authentication' :                   # if not Authentication, pass
        break
    print('I received '+ str(data))
    success, img = cap.read()   
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)         #reduce the size of the image in the webcam 
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)        #convert the image on the webcam into RGB            

    facesCurFrame = face_recognition.face_locations(imgS)  #find the face location on the webcam
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)   #encode the face on the webcam to do comparison

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)    #compair user face on the webcam with the saved images
        matchIndex = np.argmin(faceDis)                                         #determine the nearest saved encoding matrix for the user's face

        if matches[matchIndex]:                                                 #if recognition is done, draw a box around the user's face and send his name to the TCP client in PT
            name = classNames[matchIndex]#.upper()
            y1,x2,y2,x1 = faceLoc
            #because we resized the image , to draw the rectangle on the original image size
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            print(name)
            msg=name
            conn.sendall(msg.encode())                                   #send employee’s name to the client

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
                  
    sleep(10)
cap.release()
cv2.destroyAllWindows()
conn.close()
