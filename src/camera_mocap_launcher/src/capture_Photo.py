import cv2
import os

save_dir = 'calib_images'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

print("Press 'c' to capture and save a photo for calibration.")
print("Press 'q' to quit.")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('Capture Calibration Photo', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        img_path = os.path.join(save_dir, f'calib_{img_count:03d}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        img_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
