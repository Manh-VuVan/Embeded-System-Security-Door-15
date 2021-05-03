
import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np 
import mysql.connector
# import speech_recognition
import pyttsx3


window = tk.Tk()
# door_ear = speech_recognition.Recognizer()
# door_mouth = pyttsx3.init()
# passw_open = "open"
# with speech_recognition.Microphone() as mic:
#     audio = door_ear.listen(mic)
# try:
#     you = door_ear.recognize_google(audio)
# except:
#     you = " "

window.title("face recognition system")

L1 = tk.Label(window, text = "Name", font = ("Arria", 15))
L1.grid(column = 0, row = 0)
T1 = tk.Entry(window, width = 50, bd = 10)
T1.grid(column = 1, row = 0)

# L2 = tk.Label(window, text = "Age", font = ("Arria", 20))
# L2.grid(column = 0, row = 1)
# T2 = tk.Entry(window, width = 50, bd = 10)
# T2.grid(column = 1, row = 1)

# L3 = tk.Label(window, text = "Address", font = ("Arria", 20))
# L3.grid(column = 0, row = 2)
# T3 = tk.Entry(window, width = 50, bd = 10)
# T3.grid(column = 1, row = 2)

def train_classifier():
    # tạo file data trước bên ngoài
    data_dir = "F:/Program Files/VS code _ python/data"
    path = [os.path.join(data_dir,m) for m in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    messagebox.showinfo("result", " Trainning dataset completed")

B1 = tk.Button(window,text = "Tranning", font = ("Arria",15), bg = "orange", fg = 'red',command = train_classifier)
B1.grid(column = 0, row = 2)

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

        coords = []

        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "", database = "Authorized_user" )
            mycursor = mydb.cursor()
            
            mycursor.execute("select name from faces_table where id = " +str(id))
            name = mycursor.fetchone()
            name = ''+''.join(name)
    
            if confidence > 80:
                cv2.putText(img,name + "  " + str(confidence), (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
                # if you == passw_open:
                #     res = "door was open"
                # else:
                #     res = "try again"
                # door_mouth.say(res)
                # door_mouth.runAndWait()
            else:
                cv2.putText(img, "unknow" + "   " + str(confidence), (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255) , 1, cv2.LINE_AA)
            
            coords = [x,y,w,h]
        return coords
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img
    # tạo file tên là DT,  vào thư mục C:\Users\admim\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data
    # copy mục haarcascade_frontalface_default.xml  vào file DT
    faceCascade  = cv2.CascadeClassifier("DT/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img,clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    
B2 = tk.Button(window,text = "Detect the face", font = ("Arria",15), bg = "green", fg = 'white', command = detect_face)
B2.grid(column = 1, row = 2)

# cài xampp 
def generate_dataset():
    if (T1.get() == "" ):
        messagebox.showinfo("result", "Please provide complete details of the user")
    else:
        mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "", database = "Authorized_user" )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * from faces_table")
        myresult = mycursor.fetchall()
        id = 1
        for x in myresult:
            id+=1
        sql = "insert into faces_table(Id, Name) values (%s,%s)"
        val = (id, T1.get())
        mycursor.execute(sql,val)
        mydb.commit()
    
        faces_classifier = cv2.CascadeClassifier("DT/haarcascade_frontalface_default.xml")
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faces_classifier.detectMultiScale(gray,scaleFactor =1.3, minNeighbors = 5)
            if faces is():
                return None
            for (x,y,w,h) in faces:
                cropper_face = img[y:y+h, x:x+w]
            return cropper_face

        cap = cv2.VideoCapture(0)
        img_id = 0
        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200,200)) 
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user." +str(id)+ '.' +str(img_id)+ ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(400 - img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2 )

                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or int(img_id)==400:
                    break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("result", " Generating dataset completed")

B3 = tk.Button(window,text = "Generate dataset", font = ("Arria",15), bg = "pink", fg = 'blue', command = generate_dataset)
B3.grid(column = 2, row = 2)

window.geometry("700x200")
window.mainloop()