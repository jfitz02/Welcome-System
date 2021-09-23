import face_recognition
import cv2
import os
import numpy as np
import pickle


def get_encoding_for(person):
    pictures = os.listdir("./data2/"+person)
    for pic in pictures:
        print(pic)
        img = face_recognition.load_image_file("./data2/"+person+"/"+pic)
        enc = face_recognition.face_encodings(img)
        if enc == []:
            continue
        return enc[0]


folders = os.listdir("./data2")
all_encodings = {}
for folder in folders:
    all_encodings[folder] = get_encoding_for(folder)
    if all_encodings[folder] == []:
        print("Failed to get: "+folder)
        exit()
    print("Got "+folder)

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_encodings, f)
        
