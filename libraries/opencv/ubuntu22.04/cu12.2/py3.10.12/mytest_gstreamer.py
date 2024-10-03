import cv2

def main():
    unparsed_source = "rtsp://admin:admin@192.168.1.119:554/live/ch1?token=d2abaa37a7c3db1137d385e1d8c15fd2"
    target_shape = (360, 640)
    # uridecodebin uri=rtsp://admin:admin@192.168.1.118:554/live/ch1?token=d2abaa37a7c3db1137d385e1d8c15fd2 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw,format=BGRx ! videorate ! video/x-raw,framerate=7/1 ! videoconvert ! video/x-raw, format=BGR, width=640, height=360 ! appsink drop=1
    source = (
        f"uridecodebin uri={unparsed_source} ! videoconvert ! "
        f"videoscale ! video/x-raw,width={target_shape[1]},height={target_shape[0]} ! "
        "appsink"
    )

    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: VideoCapture not opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()