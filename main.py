import face_recognition
import cv2
import os
import numpy as np
import pickle

# all_encodings={}


# def get_encoding_for(person):
#     pictures = os.listdir("./data2/"+person)
#     for pic in pictures:
#         print(pic)
#         img = face_recognition.load_image_file("./data2/"+person+"/"+pic)
#         enc = face_recognition.face_encodings(img)
#         if enc == []:
#             continue
#         return enc


# josh_face_encoding = get_encoding_for("Josh")[0]
# print("GOT JOSH")
# pigeon_face_encoding = get_encoding_for("Pigeon")[0]
# print("GOT PIGEON")
# michal_face_encoding = get_encoding_for("Michal")[0]
# print("GOT MICHAL")
# nathan_face_encoding = get_encoding_for("Nathan")[0]
# print("GOT NATHAN")

# all_encodings["Josh"] = josh_face_encoding
# all_encodings["Pigeon"] = pigeon_face_encoding
# all_encodings["Michal"] = michal_face_encoding
# all_encodings["Nathan"] = nathan_face_encoding

# with open('dataset_faces.dat', 'wb') as f:
#     pickle.dump(all_encodings, f)

with open('dataset_faces.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

face_names = list(all_face_encodings.keys())
face_encodings = np.array(list(all_face_encodings.values()))

#test = face_recognition.load_image_file("./data2/Pigeon/20210914_165754_030.jpg")
test = face_recognition.load_image_file("test.jpg")
face_encoding = face_recognition.face_encodings(test)[0]

match = face_recognition.compare_faces(face_encodings, face_encoding)

face_distances = face_recognition.face_distance(face_encodings, face_encoding)
best_match_index = np.argmin(face_distances)
print(face_distances)
name = "Unknown"
if match[best_match_index] and face_distances[best_match_index]<0.4:
    name = face_names[best_match_index]

print(name)