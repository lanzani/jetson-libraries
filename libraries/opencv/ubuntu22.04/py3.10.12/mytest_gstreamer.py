import cv2


def main():
    unparsed_source = "rtsp://admin:admin@192.168.1.119:554/live/ch1?token=d2abaa37a7c3db1137d385e1d8c15fd2"
    target_shape = (360, 640)

    source = (
        f"rtspsrc location={unparsed_source} latency=200 ! queue ! rtph264depay ! h264parse ! "
        f"nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR, width={target_shape[1]}, height={target_shape[0]} ! "
        f"queue ! appsink max-buffers=60 drop=true"
    )

    cap = cv2.VideoCapture(unparsed_source)

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
