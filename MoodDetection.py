import cv2
import numpy as np
from keras.models import model_from_json
import time
import mysql.connector


# Connect to MySQL
try:
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="detection"
    )
    print("Connected to MySQL")
except mysql.connector.Error as err:
    print(f"Error: {err}")

# Create a cursor object
cursor = db_connection.cursor()

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

time_interval = 2  # save frame every 30 seconds
start_time = time.time()

# Function to save the detected values to the database
def save_to_database(cursor, emotion_index):
    try:
        # Assuming you have a table named 'emotions' with columns 'id' and 'emotion'
        sql = "INSERT INTO emotions (emotion) VALUES (%s)"
        val = (emotion_dict[emotion_index],)
        cursor.execute(sql, val)
        print("Detected emotion saved to the database.")
        # Commit changes and close the connection
        db_connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")


while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Check if it's time to save the detected values
        if time.time() - start_time >= time_interval:
            # Save the detected values to the database
            save_to_database(cursor, maxindex)
            
            # Update the start time for the next interval
            start_time = time.time()

    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

db_connection.close()
cap.release()
cv2.destroyAllWindows()
