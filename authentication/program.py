import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        raise ValueError(f"No face found in image {image_path}. Please use a different image or check the file.")
    return encodings[0]

# Load reference images and get their encodings
try:
    akash_encoding = get_face_encoding("photos/akash.jpg")
except ValueError as e:
    print(e)
    exit(1)

try:
    bhuvan_encoding = get_face_encoding("photos/bhuvan.jpg")
except ValueError as e:
    print(e)
    exit(1)

known_face_encodings = [akash_encoding, bhuvan_encoding]
known_face_names = ["Akash", "Bhuvan"]

attendance = set()
video_capture = cv2.VideoCapture(0)

with open('Attendance.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Name", "Time", "Date"])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing and convert from BGR to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Use a stricter tolerance value
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            # Check if the best match is valid and below the threshold
            if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

            face_names.append(name)

            # Mark attendance only once per recognized face
            if name != "Unknown" and name not in attendance:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                current_date = now.strftime("%Y-%m-%d")
                csv_writer.writerow([name, current_time, current_date])
                attendance.add(name)
                print(f"{name} marked present at {current_time} on {current_date}")

        # Draw bounding boxes and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale up the face locations since the frame was resized
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Attendance System', frame)

        # Exit the loop when key '1' is pressed
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

video_capture.release()
cv2.destroyAllWindows()
