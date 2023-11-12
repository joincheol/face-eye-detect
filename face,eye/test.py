import cv2

# OpenCV의 얼굴 및 눈 감지기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 이미지 읽기
image_path = "sim2.jpg"  # 이미지 입력
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 이미지를 그레이스케일로 변환

# 얼굴 감지
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# 각 얼굴에 대해 루프 수행
for (x, y, w, h) in faces:
    # 얼굴 윤곽선을 그림
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 눈 감지
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)

# 결과 표시
cv2.imshow("Detection", image) # Detection 이란 제목의 창에 결과 저장
cv2.waitKey(0) # 키 입력시 결과 화면 종료
cv2.destroyAllWindows() 
