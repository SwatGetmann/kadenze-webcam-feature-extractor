import cv2
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

# try https://github.com/ageitgey/face_recognition

current_milli_time = lambda: int(round(time.time() * 1000))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

videoCapture = cv2.VideoCapture(0)
frameWidth = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

wekinator_client = udp_client.SimpleUDPClient(
    address='127.0.0.1',
    port=6448
)

fx, fy, fw = 0, 0, 0

while True:
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        for (i, (x, y, w, h)) in enumerate(faces):
            fx, fy, fw = x, y, w

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 2)
            roi_gray = gray[y:(y + h), x:(x + w)]
            roi_color = frame[y:(y + h), x:(x + w)]

            eyes = eyeCascade.detectMultiScale(
                roi_gray
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

            cv2.putText(
                img=frame,
                text="Face[{}] [x={},y={},w={},h={},height={}]".format(
                    i,
                    x,
                    y,
                    w,
                    h,
                    frameHeight - y
                ),
                org=(10, 32 + i * 32),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0)
            )
    else:
        fx, fy, fw = 0, 0, 0

    print("fx = {}, fy = {}, fw ={}".format(fx, fy, fw))

    wekinator_client.send_message(
        address='/wek/inputs',
        value=[float(fx), float(fy), float(fw)]
    )

    cv2.imshow('Web Cam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()