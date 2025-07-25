import cv2
import numpy as np
import glob

# Checkerboard parameters (GP500-35-12 * 9)
CHECKERBOARD = (11, 8)  # Inner corners count (columns, rows) = (12-1, 9-1)
SQUARE_SIZE = 35  # 35mm per square

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (world coordinates)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

# Load calibration images (adjust path as needed)
images = glob.glob('calibration_images/*.jpg')  # Should contain 15-20 images[9](@ref)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        objpoints.append(objp)
        
        # Refine corner locations to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_refined)
        
        # Draw and display the corners (optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration
if len(imgpoints) == 0:
    print("No valid checkerboard images found. Calibration cannot proceed.")
    exit(1)
img_shape = gray.shape[::-1]  # Use the last successfully loaded grayscale image
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

# Print calibration results
print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("\nDistortion Coefficients (k1,k2,p1,p2,k3):\n", dist.ravel())
print("\nReprojection Error:", ret)  # Lower is better (typically < 0.5)[6](@ref)

# Save calibration data (optional)
np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

# Undistort test image (optional)
test_img = cv2.imread('test_image.jpg')
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
cv2.imwrite('calibrated_result.jpg', dst)