# Marc Badger
# 9.14.17
# Advanced Land Lines Project

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nx = 9
ny = 6

save_undist_images = False

# first prepare the locations of the corners in the checkerboard coordinate system: (0,0,0), (1,0,0)...
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# We will store object points and image points for all images in an array
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points on the image plane

# get a list of the images in the calibration folder
images = glob.glob('./camera_cal/calibration*.jpg')

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # findChessboardCorners needs a grayscale image

	# Use OpenCV's built-in function to find chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	# If we found a chesboard, set the object points and image points
	if ret == True:
		print ('working on ', fname)
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw the checkerboard corners on the image
		cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		# Save the image for the writeup
		write_name = './output_images/checkerboard_corners/corners_found'+str(idx)+'.jpg'
		cv2.imwrite(write_name, img)

# load an image so we can get it's dimensions
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Now use OpenCV's built in calibration routine.  Takes in:
# objpoints: 3D locations of points in local checkerboard coordinates
# imgpoints: corresponding 3D locations of these points in image coordinates
# img_size: the dimensions of the image
# Note: right now, we do not care about absolute scale, so we don't need to put
#		in the square size of our checkerboard
# Outputs:
# ret:
# mtx: 3x3 camera matrix with focal lengths and centers of projection
# dist: distortion coefficients k1, k2, p1, p2, [k3, k4, k5, k6]
# rvecs: rotation vectors of checkerboards relative to camera
# tvecs: translation vectors of checkerboards relative to camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# save the results as a pickel
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "./calibration_pickle.p", "wb"))

# Finally, save the undistorted figures

# The following function is from the lesson material:
# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # we need something to return even if we didn't find anything
    else:
    	M = 1
    	warped = 1

    # Return the resulting image and matrix
    return ret, warped, M

if save_undist_images:
	for idx, fname in enumerate(images):
		img = cv2.imread(fname)

		ret, top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
		if ret == True:
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
			f.tight_layout()
			ax1.imshow(img)
			ax1.set_title('Original Image', fontsize=50)
			ax2.imshow(top_down)
			ax2.set_title('Undistorted and Warped Image', fontsize=50)
			plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

			# Save the image for the writeup
			write_name = './output_images/checkerboard_corners/undistorted_corners_found'+str(idx)+'.jpg'
			plt.savefig(write_name, bbox_inches='tight')