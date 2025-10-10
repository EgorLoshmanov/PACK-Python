import cv2

def main() -> None:
    cap = cv2.VideoCapture("tasks/pack5/edit.mp4")
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = cv2.waitKey(15) & 0xff
        cv2.imshow('Frame', frame)
        if key == 27: # Esc
            break
            
    cv2.destroyWindow('Frame')
    cap.release()

if __name__ == "__main__":
    main()