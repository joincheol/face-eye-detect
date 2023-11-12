import cv2

# OpenCV의 얼굴 및 눈 감지기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 웹캠에서 비디오를 캡처
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냅니다.

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 각 얼굴에 대해 루프 수행
    for (x, y, w, h) in faces:
        # 얼굴 윤곽선을 그림
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 눈 감지
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)

    # 결과 프레임을 화면에 표시
    cv2.imshow("Facial Features Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
