import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error check your cammera or input chanel")
        break
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotions = result[0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
    except Exception as e:
        dominant_emotion = "Not Detected"

    cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
