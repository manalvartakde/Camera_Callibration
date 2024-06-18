import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

'''
Description of the code:
Checker board corners are detcted and we can get the camera callibration matrix only for single camera setup
'''



# Termination criteria for corner subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,9,0) for a 9x10 board
objp = np.zeros((9*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:10].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

image_dir = 'C:/Users/manal/Documents/Python/Camera_Callibration/checkerboard_Images'  
images = glob.glob(os.path.join(image_dir, '*.jpg'))
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 10), None)
    
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 10), corners2, ret)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # Save the camera parameters for future use
# np.savez('calibration_data.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Undistortion example
# img = cv2.imread('path_to_your_test_image.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibrated_result.jpg', dst)

print("Camera matrix:")
print(mtx)
print("Distortion coefficients:")
print(dist)
