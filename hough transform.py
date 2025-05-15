import cv2
import numpy as np

# 啟動攝影機（預設裝置編號為 0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取畫面")
        break

    # 轉灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 套用高斯模糊，減少雜訊（建議步驟）
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用 Otsu 二值化（主要是參考，可視情況使用）
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用 Hough Transform 偵測圓形（改用模糊後圖像）
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=35,
        minRadius=20,
        maxRadius=60
    )
    distance = 10 # 現實距離 單位cm
    # 畫出圓形
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            print("x : ", i[0], ", y : ", i[1])

    # 顯示結果畫面
    cv2.imshow('Hough Circle Detection', frame)
    

    # 按 'q' 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
