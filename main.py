import face_recognition
import cv2
import os
import numpy as np
import pickle

import pigpio # Allows pi to control the pins!

pi = pigpio.pi()

colours = {} # Dictionary containing preferred colours of each user
RED_PIN = 0 # The pin numbers of the red pin
GREEN_PIN = 0
BLUE_PIN = 0


with open("colours.txt") as f: # Read the users' colour preferences from the colours.txt settings file
    lines = f.readlines() # Settings should be formatted as "name red_value green_value blue_value"
    for line in lines:
        name, red, green, blue = line.split(' ')
        colours[name] = (int(red), int(green), int(blue))
        


def change_lights(name: str): # Change the colour of the lights to the name's preferred colour
    colour = colours.get(name, (255, 255, 255)) # If you can't get the options for a name, then default to white
    pi.set_PWM_dutycycle(RED_PIN, colour[0])
    pi.set_PWM_dutycycle(GREEN_PIN, colour[1])
    pi.set_PWM_dutycycle(BLUE_PIN, colour[2])




with open('dataset_faces.dat', 'rb') as f:
 	all_face_encodings = pickle.load(f)



face_names = list(all_face_encodings.keys())
face_encodings = np.array(list(all_face_encodings.values()))


video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    cv2.imshow('Video', rgb_small_frame)
    cv2.imshow('Video2', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    face_locations = face_recognition.face_locations(rgb_small_frame)
    try:
        face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_locations[0]])[0]
    except IndexError as e:
        print("Could not find anyone")
        continue

    match = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "unknown"

    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if match[best_match_index] and face_distances[best_match_index]<0.5:
        name = face_names[best_match_index]

    print(name)

    change_lights(name) # Change the lights to the preferred colour of the person with given name

pi.stop() # Stop the pigpio stuff when this script ends
