import cv2
import time
import mediapipe as mp
import math
import pyautogui

cap = cv2.VideoCapture(0)  # 0 for default webcam
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera.")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Get the coordinates of points for face width
            point_132 = faceLms.landmark[132]
            point_361 = faceLms.landmark[361]

            # Calculate the face width
            face_width = math.sqrt((point_132.x - point_361.x)**2 + (point_132.y - point_361.y)**2 + (point_132.z - point_361.z)**2)

            # Get the coordinates of point 57 and 0
            point_57 = faceLms.landmark[216]
            point_0 = faceLms.landmark[0]

            # Calculate the distance between point 57 and 0
            distance_mouth = math.sqrt((point_57.x - point_0.x)**2 + (point_57.y - point_0.y)**2 + (point_57.z - point_0.z)**2)
            proportional_distance_mouth = distance_mouth / face_width

            # Get the coordinates of points 12 and 14
            point_12 = faceLms.landmark[12]
            point_14 = faceLms.landmark[14]

            # Calculate the distance between points 12 and 14
            distance_12_14 = math.sqrt((point_12.x - point_14.x)**2 + (point_12.y - point_14.y)**2 + (point_12.z - point_14.z)**2)
            proportional_distance_12_14 = distance_12_14 / face_width

            # Get the coordinates of points 287 and 0
            point_287 = faceLms.landmark[436]

            # Calculate the distance between points 287 and 0
            distance_287_0 = math.sqrt((point_287.x - point_0.x)**2 + (point_287.y - point_0.y)**2 + (point_287.z - point_0.z)**2)
            proportional_distance_287_0 = distance_287_0 / face_width

            # Get the coordinates of points 159 and 145
            point_52 = faceLms.landmark[52]
            point_145 = faceLms.landmark[145]

            # Calculate the distance between points 159 and 145
            distance_52_145 = math.sqrt((point_52.x - point_145.x)**2 + (point_52.y - point_145.y)**2 + (point_52.z - point_145.z)**2)
            proportional_distance_159_145 = distance_52_145 / face_width

            # Draw the lines
            point_57_x, point_57_y = int(point_57.x * img.shape[1]), int(point_57.y * img.shape[0])
            point_0_x, point_0_y = int(point_0.x * img.shape[1]), int(point_0.y * img.shape[0])
            cv2.line(img, (point_57_x, point_57_y), (point_0_x, point_0_y), (0, 255, 0), 2)

            point_12_x, point_12_y = int(point_12.x * img.shape[1]), int(point_12.y * img.shape[0])
            point_14_x, point_14_y = int(point_14.x * img.shape[1]), int(point_14.y * img.shape[0])
            cv2.line(img, (point_12_x, point_12_y), (point_14_x, point_14_y), (0, 255, 0), 2)

            point_287_x, point_287_y = int(point_287.x * img.shape[1]), int(point_287.y * img.shape[0])
            cv2.line(img, (point_287_x, point_287_y), (point_0_x, point_0_y), (0, 255, 0), 2)

            point_52_x, point_52_y = int(point_52.x * img.shape[1]), int(point_52.y * img.shape[0])
            point_145_x, point_145_y = int(point_145.x * img.shape[1]), int(point_145.y * img.shape[0])
            cv2.line(img, (point_52_x, point_52_y), (point_145_x, point_145_y), (0, 255, 0), 2)

            # Draw the proportional distances on the image
            cv2.putText(img, f"Izquierda: {proportional_distance_mouth:.2f}", (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(img, f"Boca: {proportional_distance_12_14:.2f}", (20, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(img, f"Derecha: {proportional_distance_287_0:.2f}", (20, 220), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(img, f"Cejas: {proportional_distance_159_145:.2f}", (20, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            if proportional_distance_12_14 > 0.23:
                pyautogui.click()
            elif proportional_distance_mouth >= 0.33:
                pyautogui.scroll(-18)  # Scroll down
            elif proportional_distance_287_0 >= 0.33:
                pyautogui.scroll(18)
            elif proportional_distance_159_145 >= 0.34:
                pyautogui.press('enter')

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)