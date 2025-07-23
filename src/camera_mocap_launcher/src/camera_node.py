#!/usr/bin/env python3
import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('/dev/video0')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    fx_r = 582.59118861
    fy_r = 582.65884802
    cx_r = 629.53535406
    cy_r = 348.71988126
    k1_r = 0.00239457
    k2_r = -0.03004914
    p1_r = -0.00062043
    p2_r = -0.00057221
    k3_r = 0.01083464
    cameraMatrix_r = np.array([[fx_r,0.0,cx_r], [0.0,fy_r,cy_r], [0.0,0.0,1]], dtype=np.float32)
    distCoeffs_r = np.array([k1_r, k2_r, p1_r, p2_r, k3_r], dtype=np.float32)
    dz2 = (1280,720)
    newCameraMatrix_r, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix_r, distCoeffs_r, dz2, 0, dz2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(cameraMatrix_r, distCoeffs_r, None, newCameraMatrix_r, dz2, cv2.CV_16SC2)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to quit the camera feed")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    mouse_coords = {'x': 0, 'y': 0}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_coords['x'] = x
            mouse_coords['y'] = y

    cv2.namedWindow('Camera Feed')
    cv2.setMouseCallback('Camera Feed', mouse_callback)

    try:
        while True:
            ret, imgrgb = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            img = cv2.remap(imgrgb, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)
            # Crop the image to region (105,70) to (1196,659)
            img = img[70:660, 105:1197]
            # Draw mouse coordinates only in the window, not in terminal
            coord_text = f"({mouse_coords['x']}, {mouse_coords['y']})"
            cv2.putText(img, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Camera Feed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nCamera feed interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main()
