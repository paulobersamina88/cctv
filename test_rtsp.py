
import cv2

url = input("Paste RTSP URL: ")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open stream.")
else:
    print("Stream opened. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("RTSP Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
