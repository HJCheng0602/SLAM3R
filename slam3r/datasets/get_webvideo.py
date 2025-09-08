import cv2
import time

STREAM_URL = "http://rab:12345678@192.168.137.83:8081"

class Get_online_video:
    def __init__(self, STREAM_URL):
        self.stream_url = STREAM_URL
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            print("can not access the video stream, please check your url and camera state")
            exit()
    
    def get_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("the stream has been stopped")
            self.cap.release()
            return None 
        return frame

# --- Main Program ---
if __name__ == "__main__":
    miaomiao = Get_online_video(STREAM_URL)
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)

    while True:
        frame = miaomiao.get_video_frame()

        if frame is None:
            break
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)
    miaomiao.cap.release()
    cv2.destroyAllWindows()