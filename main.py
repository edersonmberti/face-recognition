import cv2 

haarCascadeFrontalFacePath = "haarcascade_frontalface_default.xml"
haarCascadeEyePath = "haarcascade_eye.xml"

windowName = "Output"

frameWidth = 1080
frameHeight = 1080

greenColor = (0, 255, 0)
redColor = (255, 0, 0)
blueColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

cascadeFrontalFaces = cv2.CascadeClassifier(haarCascadeFrontalFacePath)
cascadeEyes = cv2.CascadeClassifier(haarCascadeEyePath)

while True:
    success, image = cap.read()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frontalFacesDetected = cascadeFrontalFaces.detectMultiScale(grayImage)
    eyesDetected = cascadeEyes.detectMultiScale(grayImage)

    for (x, y, w, h) in frontalFacesDetected:
        cv2.rectangle(image, (x, y), (x + w, y + h), greenColor, 2)
        cv2.putText(image, "Face", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, greenColor, 2, cv2.LINE_AA)

    for (x, y, w, h) in eyesDetected:
        cv2.rectangle(image, (x, y), (x + w, y + h), redColor, 2)
        cv2.putText(image, "Eye", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, redColor, 2, cv2.LINE_AA)

    cv2.imshow(windowName, image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
