import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {
    0: "Angry contempt",
    1: "Happily disgust",
    2: "Happily surprise",
    3: "Sadly angry",
    4: "Sadly fearfull"
}

with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1),
            0
        )
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]
        cv2.putText(
            frame,
            emotion_label,
            (x + 5, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
