import cv2
import numpy as np

def main():
    image = np.zeros((480, 640, 3), dtype="uint8")

    image [:100, 150:, :] = 255
    image [200:300, :250] = (130, 50, 5) # BGR
    subimage = np.copy(image[:380, :300, :])

    image[:100, :100] = (0, 0, 255)

    fimage = image.astype("float64") #convert to double

    # uimage = cv2.convertScaleAbs(fimage) #
    uimage = np.clip(np.round(fimage), 0, 255).astype("uint8")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_channel = np.expand_dims(gray_image, axis=-1) # Add in a channel 1 helps for deep leanring
    gray_channel_only = np.expand_dims(gray_channel, axis=0)


    # cv2.imshow("Image", image)
    # cv2.imshow("SubImage", subimage)
    # cv2.imshow("fimage", fimage)
    # cv2.imshow("uimage", uimage)
    # cv2.imshow("gray_image", gray_image)
    # print(gray_image.shape)
    # cv2.imshow("gray_channel", gray_channel)
    # print(gray_channel.shape)

    # cv2.waitKey(-1)

    videocap = cv2.VideoCapture("images/noice.mp4")

    if not videocap.isOpened():
        print("HELP!")
        exit(1)
    
    key = -1
    last_frame = None
    while key == -1:
        _, frame = videocap.read()

        frame_cnt = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(videocap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_cnt == frame_index:
            videocap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cv2.imshow("NOICE", frame)
        key = cv2.waitKey(500) #30)

        if last_frame is None:
            last_frame = np.copy(frame)

        output = (frame.astype("float64") * 0.5 + last_frame.astype("float64") * 0.5)
        output = cv2.convertScaleAbs(output)

        if frame_index % 10 == 0:
            last_frame = np.copy(frame)

    videocap.release()

if __name__ == "__main__":
    main()