# Marc Badger
# 9.23.17
# Advanced Lane Finding Project
# NOTE! Much of this code is nearly an exact duplicate of the code in video_gen.py,
# 		with some minor changes for working with and saving images of intermediate steps.
# See also video_gen.py, tracker.py, and camera_calibration.py

import numpy as np
import cv2
import pickle
import glob
from tracker_vid import tracker_vid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

save_undist = True
save_undist_comp = True
save_preprocessing_pieces = True
save_warped = True
save_warped_preprocessed = True
save_windows = True
save_tracked_warped = True
save_tracked_road = True

# Read in the saved camera matrix and distortion coefficients
# See camera_calibration.py for how this file was generated
dist_pickle = pickle.load( open( "./calibration_pickle2.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

directory = './output_images'

images = glob.glob('./test_images/test*.jpg')
images.extend(glob.glob('./test_images/straight_lines*.jpg'))

########################
# Function Definitions #
########################

# abs_sobel_thresh
# Input: previously computed Sobel x and y gradients, the desired gradient orientation, threshold min & max values.
# Applies the following steps
	# 1) Take the absolute value of the derivative or gradient
	# 2) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	# 3) Create a mask of 1's where the scaled gradient magnitude 
			# is > thresh_min and < thresh_max
# Output: Return the mask as a binary_output image
def abs_sobel_thresh(sobelx, sobely, orient='x', thresh=(0,255)):
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(sobelx)
	if orient == 'y':
		abs_sobel = np.absolute(sobely)
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	# Return the result
	return binary_output

# mag_threshold
# Input: previously computed Sobel x and y gradients, threshold min & max values.
# Applies the following steps
	# 1) Calculate the magnitude 
	# 2) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	# 3) Create a binary mask where mag thresholds are met
# Output: Return this mask as your binary_output image
def mag_threshold(sobelx, sobely, mag_thresh=(0, 255)):
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output

# dir_threshold: a direction threshold for image gradients (Not currently used)
# Input: previously computed Sobel x and y gradients, a threshold angle, dir_thresh.
# Applies the following steps
	# 1) Calculate the angle
	# 2) Keeps pixes with gradient angles within dir_thresh of straight up or straight down
	#		Note: this expects a perspective transformed "bird's eye view"
	# 3) Create a binary mask where mag thresholds are met
# Output: Return this mask as your binary_output image
# NOTE: Max supression needs to be added to make this function usable!
def dir_threshold(sobelx, sobely, dir_thresh=np.pi/4):
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	# threshold is an angular distance in radians centered around pi/2 and -pi/2
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[((absgraddir >= np.pi/2-dir_thresh) & (absgraddir <= np.pi/2+dir_thresh)) | ((absgraddir >= -np.pi/2-dir_thresh) & (absgraddir <= -np.pi/2+dir_thresh))] = 1

	# Return the binary image
	return binary_output

# color_threshold: thresholds an image using the hsv color channel
# Input: a perspective transformed RGB image
# Selects regions of an image based on specified HSV ranges
# Output: a masked image of 0s and 1s representing the detected lines.
def color_threshold(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	# could also try adaptive thresholding of the image to elminate effects of large shadows
	# and pavement color changes.

	# select yellow mask
	lower_y = np.array([15, 65, 120], np.uint8) # 15, 100, 120 #[15, 80, 10]
	upper_y = np.array([120, 255, 255], np.uint8) # 80, 255, 255
	mask_y = cv2.inRange(hsv, lower_y, upper_y)

	#select yellow mask
	lower_y2 = np.array([5, 34, 113], np.uint8) # 15, 100, 120 #[15, 80, 10]
	upper_y2 = np.array([21, 74, 165], np.uint8) # 80, 255, 255
	mask_y2 = cv2.inRange(hsv, lower_y2, upper_y2)
	
	# select white mask
	lower_w = np.array([0, 0, 200], np.uint8) # 0, 0, 200
	upper_w = np.array([255,30,255], np.uint8) # 255, 30, 255 #[255,60,255]
	mask_w = cv2.inRange(hsv, lower_w, upper_w)

	# Some code to output the color channels to images
	# write_name = 'warped_hsv_h.jpg'
	# cv2.imwrite(write_name, hsv[:,:,0])
	# write_name = 'warped_hsv_s.jpg'
	# cv2.imwrite(write_name, hsv[:,:,1])
	# write_name = 'warped_hsv_v.jpg'
	# cv2.imwrite(write_name, hsv[:,:,2])
	
	# Combine the masks
	mask_yw = cv2.bitwise_or(mask_y, mask_w)

	output = np.zeros_like(image[:,:,0])
	output[(mask_yw != 0)] = 1

	# If we didn't find very many pixels of interest, add another color range that should help
	if np.sum(mask_yw) < 4500000:
		output[(mask_y2 != 0)] = 1

	return output

# region_of_interest
# Input: img an image, two lists of vertices for outer and inner polygons
# Output: a masked image containing only the pixels in between the polygons, all other pixels black
def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)

	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
		keep_mask_color = (0,) * channel_count
	else:
		ignore_mask_color = 255
		keep_mask_color = 0
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)

	return masked_image

####################################
# A loop that processes each image #
####################################

# Uncomment to only run on one or specific images
#images = [images[7]]

for idx, fname in enumerate(images):
	# read in image
	img = cv2.imread(fname)
	# Note: the movie importer gives us RGB images, so for consistency, we'll convert from BGR to RGB here
	img = img[:,:,::-1] # or could use cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	filename = fname.split('\\')[-1]

	###############################
	# undistort the image
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)

	if save_undist:
		undist_bgr = undistorted[:,:,::-1]
		write_name = directory+'/undistorted_'+filename
		cv2.imwrite(write_name, undist_bgr)

	if save_undist_comp:
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize=50)
		ax2.imshow(undistorted)
		ax2.set_title('Undistorted Image', fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		write_name = directory+'/undistorted_comparison_'+filename
		plt.savefig(write_name, bbox_inches='tight')
		plt.close()

	###############################
	# Perspective transform (following code based on the project walkthrough)
	# These values were chosen to make sure the lanes in the "straight_lines" examples
	# were as parallel as possible (the distance between lines at bottom and top should be the same).

	# Note I do the perspective transform so that sobelx will give a strong signal for lane lines

	img_height = img.shape[0]
	img_width = img.shape[1]
	bot_width = 0.60 #0.76 # percent of bottom trapizoid height
	top_width = 0.065 #0.08 # percent of top trapizoid height
	height_pct = 0.62 #0.62 # percent for trapizoid height (Controlls how far up and down you're looking on the road)
	bottom_trim = 0.935 #0.935 # percent from top to avoid car hood
	
	# define corners of the trapezoid
	top_left = [img_width*(0.5-top_width/2), img_height*height_pct] # note: vertical position down from TOP of image
	top_right = [img_width*(0.5+top_width/2), img_height*height_pct]
	bottom_left = [img_width*(0.5-bot_width/2), img_height*bottom_trim]
	bottom_right = [img_width*(0.5+bot_width/2), img_height*bottom_trim]

	src = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

	offset = img_width*0.25 # 0.25, can bump up to 0.33 if strong curve
	# dist is [top_left, top_right, bottom_right, bottom_left] to make drawing polygons easier later
	dst = np.float32([[offset, 0], [img_width-offset, 0], [img_width-offset, img_height], [offset, img_height]])

	# perform the transforms
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)

	# Get the perspective transformed "bird's eye" view
	warped = cv2.warpPerspective(undistorted,M,(img_width, img_height),flags=cv2.INTER_LINEAR)

	# Define a region of interest in the WARPED image where we'll look for lines.  The region looks like two fans coming out from
	# the bottom.
	vertices = np.array([[(223.983, 720), (223.983, 522.169), (201.997, 402.619), (141.535, \
		250.091), (78.3253, 127.794), (38.4756, 50.8427), (35.7273, 0), \
		(1280, 0), (1280, 180.011), (1148.77, 483.693), (1136.4, 585.378), \
		(1133.66, 720), (878.068, 720), (777.756, 362.77), (700.805, \
		233.602), (669.2, 170.392), (614.235, 284.444), (553.773, 425.98), \
		(509.801, 546.903), (476.822, 621.106), (460.333, 720), (223.983, 720)]], dtype=np.int32)
	warped = region_of_interest(warped, vertices)

	if save_warped:
		warped = cv2.warpPerspective(undistorted,M,(img_width, img_height),flags=cv2.INTER_LINEAR)

		warped_bgr = warped[:,:,::-1]
		write_name = directory+'/warped_undistorted_'+filename
		cv2.imwrite(write_name, warped_bgr)

		warped = region_of_interest(warped, vertices)
		warped_bgr = warped[:,:,::-1]
		write_name = directory+'/warped_roi_'+filename
		cv2.imwrite(write_name, warped_bgr)

		undist_poly = src.astype(np.int32)
		undist_poly = undist_poly.reshape((-1,1,2))
		undist_copy = undistorted.copy()
		undistorted_w_lines = cv2.polylines(undist_copy, [undist_poly], True, (255,0,0), thickness=3, lineType = cv2.LINE_AA)

		warped_poly = dst.astype(np.int32)
		warped_poly = warped_poly.reshape((-1,1,2))
		warped_copy = warped.copy()
		warped_w_lines = cv2.polylines(warped_copy, [warped_poly], True, (255,0,0), thickness=3, lineType = cv2.LINE_AA)

		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(undistorted_w_lines)
		ax1.set_title('Undistorted image with source points drawn', fontsize=25)
		ax2.imshow(warped_w_lines)
		ax2.set_title('Perspective transformed image with dest. points drawn', fontsize=25)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		write_name = directory+'/warped_comparison_'+filename
		plt.savefig(write_name, bbox_inches='tight')
		plt.close()

	###############################
	# Process image and generate a binary image with pixels of interest using a combination of
	# edge and color thresholds

	sobel_kernel_size = 15

	# Convert to grayscale
	# NOTE: THE movie image importer gives us an RGB image!
	# NOTE: an area of improvement might be to take gradients of a channel of a different color space
	gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients because the functions below need them (compute them once here to save time)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

	#sobelx = cv2.GaussianBlur(sobelx,(10,10),0) # blurring the gradiends doesn't seem to do much
	#sobely = cv2.GaussianBlur(sobely,(10,10),0)

	# create a blank image in which we'll set detections to 255
	preprocessedImage = np.zeros_like(warped[:,:,0])
	gradx = abs_sobel_thresh(sobelx, sobely, orient='x', thresh=(25,255)) # 12 is a good number (or 20-100?)
	grady = abs_sobel_thresh(sobelx, sobely, orient='y', thresh=(25,255)) # 25 is a good number


	# Could also include Sobel kernel magnitude and direction thresholds here
	gradmag = mag_threshold(sobelx, sobely, mag_thresh=(50,255)) # does not have false positives during the curve unlike gradx!
	graddir = dir_threshold(sobelx, sobely, dir_thresh=np.pi/4) # graddir does not help much

	c_binary = color_threshold(warped)

	# Combine the gradient and color binary thresholds using OR
	# There is probably a better way to combine these (like semantic segmentation using convolutional neural networks!)
	# & & no good
	# & | ok
	# | | ok too
	# | & no good
	preprocessedImage[(((gradx == 1) & (gradmag == 1)) | (c_binary == 1) )] = 255

	# do another region of interest to elminate the edge effects of the previous mask on the image gradients
	vertices2 = np.array([[(233.983, 720), (233.983, 522.169), (211.997, 402.619), (151.535, \
		250.091), (88.3253, 127.794), (48.4756, 50.8427), (45.7273, 0), \
		(1270, 0), (1270, 180.011), (1138.77, 483.693), (1126.4, 585.378), \
		(1123.66, 720), (888.068, 720), (787.756, 362.77), (710.805, \
		233.602), (669.2, 160.392), (604.235, 284.444), (543.773, 425.98), \
		(499.801, 546.903), (466.822, 621.106), (450.333, 720), (233.983, 720)]], dtype=np.int32)
	preprocessedImage = region_of_interest(preprocessedImage, vertices2)

	if save_preprocessing_pieces:
		# Stack each channel to view their individual contributions in red, green, and blue respectively
		# This returns a stack of the two binary images, whose components you can see as different colors		
		##grad_binary = np.zeros_like(gradx)
		##grad_binary[(gradx == 1)] = 1
		##mag_binary = np.zeros_like(gradmag)
		##grad_mag[(gradmag == 1)] = 1
		color_binary = np.dstack((gradmag, gradx, c_binary))
		color_binary = region_of_interest(color_binary, vertices2)

		preprepre = np.array(cv2.merge((preprocessedImage,np.zeros_like(preprocessedImage),np.zeros_like(preprocessedImage))),np.uint8) # making the original road pixels 3 color channels
		pre_to_return = cv2.addWeighted(preprepre, 0.9, warped, 0.5, 0.0)

		# Plotting thresholded images
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
		ax1.set_title('Stacked thresholds')
		ax1.imshow(color_binary*255)

		ax2.set_title('Selected regions of original warped image')
		ax2.imshow(pre_to_return)

		write_name = directory+'/preprocessed_comparison_'+filename
		plt.savefig(write_name, bbox_inches='tight')
		plt.close()
		
		write_name = directory+'/preprocessed_'+filename
		cv2.imwrite(write_name, preprocessedImage)



	###############################
	# Locate lane lines and fit a polynomial
	
	# use the tracker_vid class (see tracker_vid.py) to locate and track lines:
	# image height = 720, so window height = 80 breaks the image into 9 levels
	# margin is how far it slides the window left and right (if the road is very curvy, this will need to be higher)

	# From a measurement of a road stripe: 65 pixels long should be 3 meters -> 3 meters/65 pixels
	# From a measurement of the distance between two lines: 632 pixels should be 3.7 meters -> 3.7 meters /632 pixels

	window_width = 50 #50 #90
	window_height = 80 #80
	margin = 120 #60
	Mysmooth_factor = 1 # We only have one image, so don't do any smoothing
	road_lines = tracker_vid(Mywindow_width = window_width, Mywindow_height = window_height,
						Mymargin = margin, My_ym = 3/65, My_xm = 3.7/632, Mysmooth_factor = 1)

	# Find window centroids
	window_centroids, mean_window_centroids = road_lines.find_window_centroids(preprocessedImage)

	if save_windows:
		window_vis = road_lines.get_sliding_window_vis(warped, preprocessedImage)
		window_vis_bgr = window_vis[:,:,::-1]
		write_name = directory+'/windows_' + filename
		cv2.imwrite(write_name, window_vis_bgr)

	# Fit lines to the points we found
	lane_fits, num_frames = road_lines.find_lane_fit_parameters(im_height=warped.shape[0], fit_lines_jointly = True)

	# Get xy points for those lines
	yvals, left_fitx, right_fitx = road_lines.get_line_fit_plot_points(preprocessedImage.shape[0])

	# compute left lane, right lane, and center polygons
	left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))),np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))),np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))),np.int32)

	# draw these on the image
	road = np.zeros_like(preprocessedImage) # This image contains the lines and lane surface
	road = np.array(cv2.merge((road,road,road)),np.uint8) # This image will be added on top of the camera image first so that we'll be able to see the lane line image
	road_bkg = np.zeros_like(preprocessedImage)
	road_bkg = np.array(cv2.merge((road_bkg,road_bkg,road_bkg)),np.uint8)
	cv2.fillPoly(road,[left_lane],color=[255,0,0])
	cv2.fillPoly(road,[right_lane],color=[0,0,255])
	cv2.fillPoly(road,[inner_lane],color=[0,255,0])
	cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
	cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

	# unwarp the image back to the car's view
	road_warped = cv2.warpPerspective(road, Minv, (road.shape[1], road.shape[0]), flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (road.shape[1], road.shape[0]), flags=cv2.INTER_LINEAR)

	base = cv2.addWeighted(undistorted, 1.0, road_warped_bkg, -0.6, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 0.4, 0.0)

	ym_per_pix = road_lines.ym_per_pix # meters per pixel in y dimension, see below for how these were determined
	xm_per_pix = road_lines.xm_per_pix # meters per pixel in x dimension

	# calculate the radius of curavature
	curve_fit_cr = road_lines.find_rad_curve(preprocessedImage.shape[0])
	curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])

	# # calculate the offset of the car on the road
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_diff = (camera_center-preprocessedImage.shape[1]/2)*xm_per_pix

	side_pos = 'left'
	if center_diff <= 0:
		side_pos = 'right'

	# draw the text showing curvature, offset, and frame number
	cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
	cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

	if save_tracked_warped:
		warped = cv2.warpPerspective(undistorted,M,(img_width, img_height),flags=cv2.INTER_LINEAR)

		base_2 = cv2.addWeighted(warped, 1.0, road_bkg, -0.6, 0.0)
		result_2 = cv2.addWeighted(base_2, 1.0, road, 0.4, 0.0)
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(result_2)
		ax1.set_title('Perspective transformed image with lines', fontsize=25)
		ax2.imshow(result)
		ax2.set_title('Untransformed image with lines', fontsize=25)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		write_name = directory+'/tracked_comparison_'+filename
		plt.savefig(write_name, bbox_inches='tight')
		plt.close()

	if save_tracked_road:
		result_bgr = result[:,:,::-1]
		write_name = directory+'/tracked_' + filename
		cv2.imwrite(write_name, result_bgr)

# DONE!
exit()