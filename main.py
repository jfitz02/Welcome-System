import face_recognition
import cv2
import os
import numpy as np
import pickle



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
