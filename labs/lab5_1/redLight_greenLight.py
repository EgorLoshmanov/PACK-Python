import cv2
import time
import numpy as np

def secs_left(till):
    return int(max(0.0, till - time.monotonic()))

def main():
    # настройки таймера
    ON_SECONDS  = 5.0   # "Red light" — детекция ВКЛ
    OFF_SECONDS = 5.0   # "Green light" — детекция ВЫКЛ
    detect_on = True
    deadline = time.monotonic() + ON_SECONDS


    cap = cv2.VideoCapture("labs/lab5_1/WIN_20251015_21_28_36_Pro.mp4")
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()
    green_screen1 = np.zeros((1200, 720, 3))
    green_screen1[:, :, 1] = 255
    green_screen2 = np.zeros((1200, 720, 3))
    green_screen2[:, :, 1] = 255

    while cap.isOpened():
        now = time.monotonic()
        if now >= deadline:
            detect_on = not detect_on
            deadline = now + (ON_SECONDS if detect_on else OFF_SECONDS)


        frame_vis = frame2.copy()

        # строка статуса
        status_text = (
            f"Red light - detection ON | left: {secs_left(deadline)}s"
            if detect_on else
            f"Green light - detection OFF | left: {secs_left(deadline)}s"
        )
        cv2.putText(frame_vis, status_text, (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if detect_on:
            diff  = cv2.absdiff(frame1, frame2)                           # разница кадров
            gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)                # в градации серого
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)                     # сглаживание шума
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)   # порог
            dilated = cv2.dilate(thresh, None, iterations=3)              # расширить области

            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 400:   # отсечь мелкий шум
                  continue
                cv2.drawContours(frame_vis, [contour], -1, (0, 0, 255), 2)
                cv2.drawContours(green_screen1, [contour], -1, (0, 0, 255), 2)


        cv2.imshow("frame", frame_vis)
        cv2.imshow("map", green_screen1)

        green_screen1 = green_screen2
        green_screen2 = np.zeros((720, 1200, 3))
        green_screen2[:, :, 1] = 255
        frame1 = frame2
        ret, frame2 = cap.read()

        if (cv2.waitKey(20) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

main()