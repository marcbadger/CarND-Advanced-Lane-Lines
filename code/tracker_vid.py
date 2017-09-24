# Marc Badger
# 9.23.17
# Advanced Lane Finding Project
# See also video_gen.py, image_gen.py, and camera_calibration.py

import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Code is based on Advanced Lane Finding lesson material and walkthrough

class tracker_vid():
	# A class that handles lane finding, fitting, and some plotting
	# when starting a new instance specify all unassigned variables
	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
		# list that stores all the past (left,right) center set values used for smoothing the output
		self.recent_centers = []

		# list that stores all the past (left, right) window counts for use as weights by the fitting function
		self.recent_counts = []

		# the window pixel width of the center values, used to count pixels inside center windows to determine curve values
		self.window_width = Mywindow_width

		# the window pixel height of the center values, used to count pixels inside center windows to determine curve values
		# breaks the image into vertical levels
		self.window_height = Mywindow_height

		# The pixel distance in both directions to slide (left_window + right_window) template for searching
		self.margin = Mymargin

		self.ym_per_pix = My_ym # meters per pixel in vertical axis

		self.xm_per_pix = My_xm # meters per pixel in horizontal axis

		# The number of previous frames to incorporate in lane estimates
		self.smooth_factor = Mysmooth_factor

		# A list to keep track of all previous lane parameters
		self.current_lane_fit_params = [] # will be a list of elements like ("fit_type", params)

		# Number of frames the tracker has seen (incremented when self.find_window_centroids() is called)
		self.frames_analyzed = 0

		# Min([frames_analyzed, smooth_factor])
		self.num_frames_available = 0


	# window mask
	# Input: width, height, image_ref, center, level)
	# Selects only the region (center,level) +- (width, height) in image_ref
	# Output: a binary image with the window area = 1
	def window_mask(self, width, height, img_ref, center,level):
		output = np.zeros_like(img_ref)
		output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
		return output

	# access_default_line_params
	# Input: a string representing the type of function that will be fit
	# Output: a list or list of lists of default parameters, depending on the parameter type
	def access_default_line_params(self, param_type = "joint_no_time"):
		if param_type == "joint_no_time":
			p0 = 0, 0, 1250, -650
		elif param_type == "joint_w_time":
			p0 = 0,0,   0,0,   0, 1250,   -650
		elif param_type == "joint_w_time3":
			p0 = 0,0,0,0,   0,0,0,0,   0,0,0,1250,   -650
		elif param_type == "sep_no_time":
			p0 = ((0, 0, 600), (0, 0, 1250))
		elif param_type == "sep_w_time":
			p0 = ((0,0,   0,0,   0, 600), (0,0,   0,0,   0, 1250))
		else:
			print("Default parameters not found for that function!")
			p0 = 0, 0, 1250, -650 # If you are here, you are trying to fit the wrong function using the wrong parameters!
		return p0

	# access_previous_line_params
	# Input: a string representing the type of function that will be fit
	# Output: a list or list of lists of the most recent parameters for that fit type
	#			or if no previous parameters of that type exist, the default parameters for that fit type
	def access_previous_line_params(self, param_type = "joint_no_time"):
		current_lane_fit_params = self.current_lane_fit_params

		matching_params = [e[1] for e in current_lane_fit_params if e[0] == param_type]
		if len(matching_params) > 0:
			return matching_params[-1]
		else:
			print("No previous parameters exist! Using default parameters.")
			return self.access_default_line_params(param_type = param_type)

	# find_window_centroids: The main tracking function for finding and storing lane segment positions
	# Input: a perspective transformed binary thresholded image
	# Method: Uses convolutions of a window template and sliding windows to determine the window position at each level
	#			moving up the image from the bottom.
	# Output: Adds the detected window locations and their counts to accumulator variables devined during initilization
	# Returns: averaged window locations, but this output isn't really used except for window plotting
	def find_window_centroids(self, warped):

		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin
		window_centroids = [] # Store the (left, right) window centroid positions per level
		window_counts = [] # Store the number of pixels seen by each of left and right windows per level
		window = np.ones(window_width) # Create our window template that we will use for convolutions

		# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
		# and then np.convolve the vertical image slice with the window template

		# Sum quarter bottom of the image to get the slice, could use a different ratio
		# pick 3rd quarder bottom of the image and squash all the pixels together to a 1D signal for both the left and the right
		# only consider the left portion of the image beginning:half_width
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_conv = np.convolve(window,l_sum)
		l_center = np.argmax(l_conv)
		l_counts = l_conv[l_center]
		l_center = l_center-window_width/2
		# half_width:right
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_conv = np.convolve(window, r_sum)
		r_center = np.argmax(r_conv)
		r_counts = r_conv[r_center]
		r_center = r_center-window_width/2+int(warped.shape[1]/2)
	
		# Add what we found for the first layer
		window_centroids.append((l_center,r_center))
		window_counts.append((l_counts+1, r_counts+1))

		num_levels = (int)(warped.shape[0]/window_height)

		# Go through each layer looking for max pixel locations
		for level in range(1,num_levels):
			# convolve the window into the vertical slice of the image
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)

			# Find the best left centroid by using past left center as a reference
			# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
			offset = window_width/2
			l_min_index = int(max(l_center+offset-margin,0))
			l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])
			l_counts = (conv_signal[l_min_index:l_max_index])[l_center]
			l_center = l_center+l_min_index-offset
			if l_counts < 20:	# if we didn't find anything
				if len(window_centroids) >= 1:	# do we have any windows at all?
					l_center = window_centroids[-1][0] # if so, return the window position of the previous layer
					# the above has the effect of placing the window straight moving up the image from the last detection
				else:
					l_center = 600 # otherwise, just guess
			# Find the best right centroid by using past right center as a reference
			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])
			r_counts = (conv_signal[r_min_index:r_max_index])[r_center]
			r_center = r_center+r_min_index-offset
			if r_counts < 20:
				if len(window_centroids) >= 1:
					r_center = window_centroids[-1][1] # if we didn't find anything, return the window position of the previous layer
				else:
					r_center = 1250
			# Add what we found for that layer
			window_centroids.append((l_center,r_center))
			window_counts.append((l_counts+1, r_counts+1))

		# backup statements in case everything breaks, but not really needed
		# if len(window_centroids) == 0:
		# 	window_centroids = [(600,1250)] * num_levels

		self.recent_centers.append(window_centroids)
		self.recent_counts.append(window_counts)

		# increase the frame counter for writing frame number overlay
		self.frames_analyzed += 1

		# returns an average of the smooth_factor most recent centers (not used except for plotting):
		return self.recent_centers[-1], np.average(self.recent_centers[-self.smooth_factor:], axis = 0)

	#a function for fitting lane lines to window center points with lane lines fit independently
	def road_line_sep_poly(self, X, a, b, c):
		x, t = X
		return a*x*x + b*x + c

	#a function for fitting lane lines to window center points with lane lines fit independently
	def road_line_sept_poly(self, X, a_i, a_j, b_i, b_j, c_i, c_j):
		x, t = X
		return (a_i*t + a_j)*x*x + (b_i*t + b_j)*x + (c_i*t + c_j)

	# a function for fitting lane lines to window center points when there is only one sample:
	def road_line_poly(self, X, a, b, c, line_sep):
		x, t, line_ind = X
		return a*x*x + b*x + c + line_sep*line_ind

	# a function for fitting lane lines to window center points when there are a few samples:
	def road_line_time_poly(self, X, a_i, a_j, b_i, b_j, c_i, c_j, line_sep_i):
		x, t, line_ind = X
		return (a_i*t + a_j)*x*x + \
			(b_i*t + b_j)*x + \
			(c_i*t + c_j) + \
			(line_sep_i)*line_ind

	#### LINE_SEP SHOULDN"T HAVE SO MANY HIGH PARAMETERS, IT SHOULD JUST BE CONSTANT.
	# a function for fitting lane lines to window center points when there are many samples:
	def road_line_time3_poly(self, X, a_i, a_j, a_k, a_l, b_i, b_j, b_k, b_l, c_i, c_j, c_k, c_l, line_sep_i):
		x, t, line_ind = X
		return (a_i*t*t*t + a_j*t*t + a_k*t + a_l)*x*x + \
			(b_i*t*t*t + b_j*t*t + b_k*t + b_l)*x + \
			(c_i*t*t*t + c_j*t*t + c_k*t + c_l) + \
			(line_sep_i)*line_ind

	# find_line_fit: a function for jointly fitting two lines to the recent window centers
	# Called by: find_lane_fit_parameters, which is a shell function that calls either find_lane_fit or find_independent_lane_fit
	# Input: image height (needed to determine the number of levels)
	# Method: Extracts recent window centers from self.window_centers, formats the centers for fitting
	#			Eliminates centers that have counts with very few pixels (this feature can be turned off
	#			and doesn't actually seem to matter much.)
	#			Performs an appropriate fit according to a schedule based on how much data is available
	# Output: returns list of (fit type, list of fit parameters)
	def find_line_fit(self, im_height):
		window_width = self.window_width
		window_height = self.window_height
		window_centroids = self.recent_centers[-self.smooth_factor:]
		window_counts = self.recent_centers[-self.smooth_factor:]

		num_frames = len(window_centroids)

		## ALL OF THIS BIG SECTION is just extracting the data in recent_centers and putting
		#  it into a format that works for fitting
		all_vert_poss = []
		all_ts = []
		all_left_indicator = []
		all_datay = []
		all_sigmas = [] # counts will be weights for the fit, represented as sigma = 1/weight
		for t in range(0, num_frames):
			# points used to find the left and right lines
			rightx = []
			leftx = []

			rightc = []
			leftc = []

			# If we found any window centers
			if len(window_centroids[t]) > 0:

				# Go through each level and extract the window coordinates
				for level in range(0,len(window_centroids[t])):
					# add center value found in frame to the list of lane points per left, right
					leftx.append(window_centroids[t][level][0])
					rightx.append(window_centroids[t][level][1])

					rightc.append(window_counts[t][level][0])
					leftc.append(window_counts[t][level][1])

				res_yvals = np.arange(im_height-(window_height/2),0,-window_height)

				vert_poss = np.concatenate((res_yvals, res_yvals), axis = 0)
				ts = np.ones_like(vert_poss)*t
				left_indicator = np.concatenate((np.ones_like(res_yvals), np.zeros_like(res_yvals)), axis=0)
				datay = np.concatenate((leftx, rightx), axis=0)
				weights = np.concatenate((leftc, rightc), axis=0)

				# remove entries for which counts are < 3 (i.e. where the box didn't find anything, or found something small)
				missingLevels = np.ones_like(weights)
				missingLevels[weights<=3] = 0
				# t weights the more recent observations more heavily
				sigmas = [1.0*(num_frames - t + 1)/(np.log(float(x))+1.0) for x in weights] # note I'm more heaviliy weighting recent observations
				sigmas = [1/(x+1) for x, m in zip(weights, missingLevels) if m == 1]
				# if sum(sigmas) == 0:
				# 	sigmas = np.ones_like(sigmas)
				vert_poss = [x for x, m in zip(vert_poss, missingLevels) if m == 1]
				left_indicator = [x for x, m in zip(left_indicator, missingLevels) if m == 1]
				datay = [x for x, m in zip(datay, missingLevels) if m == 1]

				# add the points for this level to the accumulators
				if sum(missingLevels) > 0:
					all_vert_poss.extend(vert_poss)
					all_ts.extend(ts)
					all_left_indicator.extend(left_indicator)
					all_datay.extend(datay)
					all_sigmas.extend(sigmas)


		# Now fit a function to the data:
		# One option is to fit lines jointly, where polynomials for left and right lines must have same shape, but can
		# be shifted left and right by an fitted parameter.
		# Note: we fit with x along the image height dimension and y along the image width dimension because
		# curved road lines might not be functions the other way around.

		# The data matrix for the fit will be of the form (y = horizontal position on image, x = vertical position, lane indicator)
		# leftx res_yvals 1
		# rightx res_yvals 0
		# where leftx are the leftx values and each row has 1 indicating the left lane
		# and rightx are the rightx values and each row has 0 indicating the right lane

		# A fitting schedule based on the number of frames we have to fit.
		# Uses more complex fits if we have a longer frame history:
		# If we only have a few frames, don't let the fit parameters vary with time
		if num_frames <= 4: #4: #10000:
			param_type = "joint_no_time"
			p0 = self.access_previous_line_params(param_type=param_type)
			poly_to_use = self.road_line_poly
		# If we have enough data, we can let the fit parameters vary with time
		elif num_frames <= 100000: # putting 10000 here prevents it from using higher order polynomials for the time varying parameters
			param_type = "joint_w_time"
			p0 = self.access_previous_line_params(param_type=param_type)
			poly_to_use = self.road_line_time_poly
		# Or we can get really crazy and allow the fit parameters to vary cubically with time
		else:
			param_type = "joint_w_time3"
			p0 = self.access_previous_line_params(param_type=param_type)
			poly_to_use = self.road_line_time3_poly

		# In case the fit does not work, we try it first.
		try:
			lane_fits, pcov = curve_fit(poly_to_use, 
				xdata = (all_vert_poss, all_ts, all_left_indicator), 
				ydata = all_datay, p0 = p0)
				# , sigma=all_sigmas, absolute_sigma=True
		except:
			print("Error fitting, using previous parameters.")
			lane_fits = p0

		# If the fit worked, lane_fits will be the new fit parameters
		# If it didn't work, lane_fits will be the parameters from the most recent succesfful fit
		return (param_type, lane_fits)

	# find_independent_line_fit: a function for fitting two independent lines to the recent window centers
	# Called by: find_lane_fit_parameters, which is a shell function that calls either find_lane_fit or find_independent_lane_fit
	# Input: image height (needed to determine the number of levels), line_ind: an indicator for which lane to fit (0 = right, 1 = left)
	# Method: Extracts recent window centers from self.window_centers, formats the centers for fitting
	#			Eliminates centers that have counts with very few pixels (this feature can be turned off
	#			and doesn't actually seem to matter much.)
	#			Performs an appropriate fit according to a schedule based on how much data is available
	# Output: returns list of (fit type, list of fit parameters)
	# NOTE: Much of this code repeats from above (SORRY!)
	def find_independent_line_fit(self, im_height, line_ind):
		window_width = self.window_width
		window_height = self.window_height
		window_centroids = self.recent_centers[-self.smooth_factor:]
		window_counts = self.recent_centers[-self.smooth_factor:]

		num_frames = len(window_centroids)

		## ALL OF THIS BIG SECTION is just extracting the data in recent_centers and putting
		#  it into a format that works for fitting
		all_vert_poss = []
		all_ts = []
		all_datay = []
		all_sigmas = [] # counts will be weights for the fit, represented as sigma = 1/weight
		for t in range(0, num_frames):
			# points used to find the lane line
			lanex = []
			lanec = []

			# If we found any window centers
			if len(window_centroids[t]) > 0:

				# Go through each level and extract the window coordinates
				for level in range(0,len(window_centroids[t])):
					# add center value found in frame to the list of lane points per left, right
					lanex.append(window_centroids[t][level][line_ind])
					lanec.append(window_counts[t][level][line_ind])

				res_yvals = np.arange(im_height-(window_height/2),0,-window_height)

				vert_poss = res_yvals
				ts = np.ones_like(vert_poss)*t
				datay = lanex
				weights =lanec

				# remove entries for which counts are < 3 (i.e. where the box didn't find anything, or found something small)
				# Uncomment all of the following lines in this block to eliminate windows where nothing was detected
				missingLevels = np.ones_like(weights)
				# missingLevels[weights<=40] = 0
				# t weights the more recent observations more heavily
				sigmas = [1.0*(num_frames - t + 1)/(np.log(float(x))+1.0) for x in weights] # note I'm more heaviliy weighting recent observations
				# sigmas = [1/(x+1) for x, m in zip(weights, missingLevels) if m == 1]
				# if sum(sigmas) == 0:
				# 	sigmas = np.ones_like(sigmas)
				# vert_poss = [x for x, m in zip(vert_poss, missingLevels) if m == 1]
				# left_indicator = [x for x, m in zip(left_indicator, missingLevels) if m == 1]
				# datay = [x for x, m in zip(datay, missingLevels) if m == 1]

				if sum(missingLevels) > 0:
					all_vert_poss.extend(vert_poss)
					all_ts.extend(ts)
					all_datay.extend(datay)
					all_sigmas.extend(sigmas)

		# A fitting schedule based on the number of frames we have to fit.
		# Uses more complex fits if we have a longer frame history:
		# If we only have a few frames, don't let the fit parameters vary with time
		if num_frames <= 4:
			param_type = "sep_no_time"
			p0 = self.access_previous_line_params(param_type=param_type)
			p0 = p0[line_ind]
			poly_to_use = self.road_line_sep_poly
		else:
			param_type="sep_w_time"
			p0 = self.access_previous_line_params(param_type=param_type)
			p0 = p0[line_ind]
			poly_to_use = self.road_line_sept_poly

		try:
			lane_fits, pcov = curve_fit(poly_to_use, 
				xdata = (all_vert_poss, all_ts), ydata = all_datay, p0 = p0)
				# , sigma=all_sigmas, absolute_sigma=True
		except Exception as e:
			print("Error fitting, using previous parameters.")
			print(e)
			lane_fits = p0

		return (param_type, lane_fits)
	
	# find_lane_fit_parameters: a shell function used call find_line_fit or find_independent_line_fit
	# Input: image height and boolean fit_lines_jointly that determines whether lines will be fit jointly or independently
	# Output: The lane fit parameters and number of frames analyzed so far (for plotting on the figure)
	def find_lane_fit_parameters(self, im_height, fit_lines_jointly):
		window_centroids = self.recent_centers[-self.smooth_factor:]
		num_frames = len(window_centroids)

		if fit_lines_jointly:
			lane_fits = self.find_line_fit(im_height)
			self.current_lane_fit_params.append(lane_fits)
		else:
			lane_fit_left = self.find_independent_line_fit(im_height, line_ind = 0)
			lane_fit_right = self.find_independent_line_fit(im_height, line_ind = 1)
			lane_fits = (lane_fit_left[0], (lane_fit_left[1], lane_fit_right[1]))
			self.current_lane_fit_params.append(lane_fits)
		
		self.num_frames_available = num_frames

		return lane_fits, num_frames

	# find_rad_curve: a function to generate points along a curve based on the last line fit.
	# Input: image height a
	# Output: a list of points lying along the curve (the acutal radius of curvature is calculated in video_gen.py)
	def find_rad_curve(self, im_height):
		window_width = self.window_width
		window_height = self.window_height

		ym_per_pix = self.ym_per_pix # meters per pixel in y dimension
		xm_per_pix = self.xm_per_pix # meters per pixel in x dimension

		yvals, left_fitx, right_fitx = self.get_line_fit_plot_points(im_height)

		# based on the left line
		curve_fit_cr = np.polyfit(np.array(yvals,np.float32)*ym_per_pix, np.array(left_fitx, np.float32)*xm_per_pix,2)

		return curve_fit_cr

	# get_sliding_window_vis: a function used for creating a visualization of the sliding windows
	# Input: warped, a perspective transformed image, and its thresholded version preprocessedImage
	# Output: an image with current sliding windows in green and mean sliding windows in thinner red
	def get_sliding_window_vis(self, warped, preprocessedImage):
		window_width = self.window_width
		window_height = self.window_height
		window_centroids = self.recent_centers[-1]
		mean_window_centroids = np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
		window_counts = self.recent_counts[-1]

		wp_to_return = warped.copy()
		wp_to_return[(preprocessedImage < 100)] = 0

		# If we found any window centers
		if len(window_centroids) > 0:

			# Points used to draw all the left and right windows
			l_points = np.zeros_like(wp_to_return[:,:,0])
			r_points = np.zeros_like(wp_to_return[:,:,0])

			# Go through each level and draw the windows	
			for level in range(0,len(window_centroids)):
				# Window_mask is a function to draw window areas
				l_mask = self.window_mask(window_width,window_height,wp_to_return[:,:,0],window_centroids[level][0],level)
				r_mask = self.window_mask(window_width,window_height,wp_to_return[:,:,0],window_centroids[level][1],level)
				# Add graphic points from window mask here to total pixels found
				# But also, don't plot windows that don't have pixels
				if window_counts[level][0] >= 3:
					l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
				if window_counts[level][1] >= 3:
					r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

			# Draw the results
			template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
			zero_channel = np.zeros_like(template) # create a zero color channel
			template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
			wp_to_return = cv2.addWeighted(wp_to_return, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
		 
		 # If we found any window centers
		if len(mean_window_centroids) > 0:

			# Points used to draw all the left and right windows
			l_points = np.zeros_like(wp_to_return[:,:,0])
			r_points = np.zeros_like(wp_to_return[:,:,0])

			# Go through each level and draw the windows	
			for level in range(0,len(mean_window_centroids)):
				# Window_mask is a function to draw window areas
				l_mask = self.window_mask(window_width-25,window_height,wp_to_return[:,:,0],mean_window_centroids[level][0],level)
				r_mask = self.window_mask(window_width-25,window_height,wp_to_return[:,:,0],mean_window_centroids[level][1],level)
				# Add graphic points from window mask here to total pixels found 
				l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
				r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

			# Draw the results
			template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
			zero_channel = np.zeros_like(template) # create a zero color channel
			template = np.array(cv2.merge((template,zero_channel,zero_channel)),np.uint8) # make window pixels green
			wp_to_return = cv2.addWeighted(wp_to_return, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
		 

		#wp_to_return = cv2.warpPerspective(wp_to_return, Minv, (wp_to_return.shape[1], wp_to_return.shape[0]), flags=cv2.INTER_LINEAR)
		#wp_to_return = cv2.addWeighted(wp_to_return, 1, undistorted, 0.5, 0.0)
		wp_to_return = cv2.addWeighted(wp_to_return, 1, warped, 0.5, 0.0)

		cv2.putText(wp_to_return, 'Frame number ' + str(self.frames_analyzed), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
		return wp_to_return

	# get_line_fit_plot_points: a function used for generating a list of xy points from the most recent fit
	# Used for plotting the lane and lines in video_gen.py
	# Input: image height
	# Output: yvals: a list of vertical positions along the image, left_fitx and right_fitx: a list of 
	# 		corresponding horizontal positions for the left and right lanes, respectively.
	def get_line_fit_plot_points(self, im_height):
		yvals = np.array(range(0, im_height),np.float32)
		num_frames = self.num_frames_available
		tvals = np.ones_like(yvals)*(num_frames-1)

		if len(self.current_lane_fit_params) < 1:
			print("No fit to get points from!  Do a fit first.")
			return yvals, tvals

		lane_fits = self.current_lane_fit_params[-1]
		fit_type = lane_fits[0]

		if fit_type[:3] == 'sep':
			if fit_type[:6] == 'sep_no':
				left_fitx = self.road_line_sep_poly((yvals, tvals), *lane_fits[1][0]) # np.ones_like indicates left lane
				right_fitx = self.road_line_sep_poly((yvals, tvals), *lane_fits[1][1]) # np.zeros_like indicates right lane

			else:
				left_fitx = self.road_line_sept_poly((yvals, tvals), *lane_fits[1][0]) # np.ones_like indicates left lane
				right_fitx = self.road_line_sept_poly((yvals, tvals), *lane_fits[1][1]) # np.zeros_like indicates right lane

		if fit_type[:5] == 'joint':
			if fit_type[:8] == 'joint_no':
				left_fitx = self.road_line_poly((yvals, tvals, np.ones_like(yvals)), *lane_fits[1]) # np.ones_like indicates left lane
				right_fitx = self.road_line_poly((yvals, tvals, np.zeros_like(yvals)), *lane_fits[1]) # np.zeros_like indicates right lane

			elif fit_type == 'joint_w_time':
				left_fitx = self.road_line_time_poly((yvals, tvals, np.ones_like(yvals)), *lane_fits[1]) # np.ones_like indicates left lane
				right_fitx = self.road_line_time_poly((yvals, tvals, np.zeros_like(yvals)), *lane_fits[1]) # np.zeros_like indicates right lane

			else:
				left_fitx = self.road_line_time3_poly((yvals, tvals, np.ones_like(yvals)), *lane_fits[1]) # np.ones_like indicates left lane
				right_fitx = self.road_line_time3_poly((yvals, tvals, np.zeros_like(yvals)), *lane_fits[1]) # np.zeros_like indicates right lane

		#left_fitx = road_lines.road_line_time_poly((yvals, np.ones_like(yvals)*num_frames, np.ones_like(yvals)), *lane_fits) # np.ones_like indicates left lane
		left_fitx = np.array(left_fitx, np.int32)

		#right_fitx = road_lines.road_line_time_poly((yvals, np.ones_like(yvals)*num_frames, np.zeros_like(yvals)), *lane_fits) # np.zeros_like indicates right lane
		right_fitx = np.array(right_fitx, np.int32)

		return yvals, left_fitx, right_fitx