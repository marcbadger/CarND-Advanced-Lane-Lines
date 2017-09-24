# Advanced Lane Finding Project

Overview
---
This repository contains files for the Advanced Lane Finding Project. A detailed writeup of the project is given below.

The goals / steps of this project were to:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[detectedCorners]: ./examples/corners_found4.jpg "Detected corners."
[undistortedCorners]: ./examples/undistorted_corners_found4.jpg "Original image and undistorted, perspective transformed result."
[undistortedRoad]: ./examples/undistorted_comparison_test1.jpg "Original image and undistorted example image."
[missingLaneParts]: ./examples/missing_lane_parts.JPG "Some segments of the lane in this image literally have the same HSV values as the road pixels do in some other frames."
[foundLaneParts]: ./examples/found_lane_parts.JPG "Adding another yellow threshold, used when total number of pixels detected is low helps."
[warpedPreprocessed]: ./examples/output1_preprocessed_warped.gif "Example image thresholded using edge and color thresholds."
[unwarpedPreprocessed]: ./examples/output1_preprocessed_unwarped.gif "Example image thresholded using edge and color thresholds (unwarped)."
[warpCheck]: ./examples/warped_comparison_test5.jpg "Check verifying perspective transformation is working."
[boxesGIF]: ./examples/output1_boxes.gif "Sliding window detections on a short clip."
[trackedJointGIF]: ./examples/output1_tracked_15_joint_w_time.gif "Polynomial fits to window boxes, lanes fit jointly so they have the same shape in the perspective transformed image."
[trackedSepGIF]: ./examples/output1_tracked_15_sep_w_time.gif "Polynomial fits to window boxes, lanes fit independently so they can have different shapes in the perspective transformed image."
[bouncingGIF]: ./examples/bouncing_example.gif "The perspective transform changes when the car bounces, making decreasing the fit quaility when fitting lines jointly."
[finalResult]: ./examples/output1_tracked_long_15_joint_w_time.gif "Final result, lines fit jointly, allowing parameters to vary linearly over time, using the previous 15 frames of window locations for each frame."
[video1]: ./project_video_tracked.mp4 "Video"
[challengeResult]: ./examples/challenge_tracked.gif "My pipeline didn't do as well on the challenge video!"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. My project includes the following files:
* [camera_calibration.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/camera_calibration.py) - a Python script to generate camera matrix and distortion coefficients from checkerboard images
* [image_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/image_gen.py) - a Python script to run the tracking pipeline on sample images
* [video_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/video_gen.py) - a Python script to run the tracking pipeline on sample videos
* [tracker_vid.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/tracker_vid.py) - a Python class that handles lane finding and fitting
* [project_video_tracked.mp4](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/project_video_tracked.mp4) - the output of the lane finding pipeline on the sample video.
* [example output images](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/tree/master/output_images) - intermediate output of image_gen.py
* [README.md, this document](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/README.md) - a writeup report summarizing the results

### Camera Calibration

OpenCV makes it a sinch to calculate a camera matrix and distortion parameters from a list of checkerboard images (e.g., "Original Image" in the figure below).  The code for this camera calibration is contained in camera_calibration.py.

The first step uses the `cv2.findChessboardCorners()` function to detect strong corners in an image and fit to these points a checkerboard of a specified number of squares. The image below shows an example of detected corners:

![alt text][detectedCorners]

Next, I used `cv2.calibrateCamera()` function to fit a 3D set of checkerboard points to the detected image points (code lines 51-69 in camera_calibration.py).  The function takes in a set of 3D locations of the corner points in local checkerboard coordinates (i.e. z = 0 for all points), a set of 2D pixel locations of the corresponding corner points in image coordinates and determines the rotation and translation vector of the checkerboard relative to the camera so that the reprojection error for the estimated 3D world coordinates of the checkerboard points is minimized.  We save the camera matrix and undistortion coefficients for later use (code lines 65-69 in camera_calibration.py)

Finally, I used the calculated camera matrix (which contains the focal lengths and centers of projection) and undistortion coefficients to correct for lense distortion (code lines 77-78 in camera_calibration.py) and compute a perspective transformation (using the `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions inside my `corners_unwarp()` function) that shows the checkerboard from straight ahead (code lines 73-113 in camera_calibration.py).  The undistorted, perspective transformed result is shown below:

![alt text][undistortedCorners]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

With our precomputed camera matrix and distortion coefficients, we can now undistort any image using `cv2.Undistort()`. In the image below you can see that this makes the slightly curving guard rail slightly straighter at the edges of the video (some of the original image is clipped by the undistortion function).

![alt text][undistortedRoad]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In my pipeline, I performed a perspective transform first (details below).  I did so because I expected the Sobel x gradient to be even more helpful in identifying lines in the "bird's eye view" because straight lines would now go straght up and down the image.

Determining an appropriate color transform was the hardest part of the project and really made me wish I'd chosen to try doing semantic segmentation using FCNs from the start (e.g. [Long, et al. 2015](https://arxiv.org/pdf/1411.4038.pdf)), or at least spent time developing a GUI with which to sample color ranges based on region selections.  The task is to select target pixels using color and edge thresholds.  My color thresholds used the HLS (hue, saturation, lightness) color space.  Similar to my approach on [the first lane finding project](https://github.com/marcbadger/CarND-LaneLines-P1), I found yellow lines using the `cv2.inRange()` function to create a mask with a HLS intensity range of (15-120, 65-255, 120-255).  Some yellow lines had saturation less than 65, but decreasing the threshold on the saturation channel too much caused large sections of the road to be detected.  I found white lines using a range of (0-255, 0-30, 200-255).  I combined yellow and white masks using an OR operation.

It turns out that the distribution of HLS intensities of the lines in some frames overlaps with that of the road in other frames (meaning that a single color threshold could not separate the line in all frames). For example the pixels circled below are literally the same HSV color as the road in several other frames.

![alt text][missingLaneParts]

In cases where the number of detected pixels was below a certain threshold, I supplemented the yellow and white detections with an additional range (5, 34, 113) to (120, 255, 255) (code lines 109-112, 134-136 in video_gen.py). Doing so helped it find the additional part of the lane line at the cost of some noise in the bottom right.

![alt text][foundLaneParts]

I found gradient thresholds using the approach in the project description.  On perspective transformed images, I found Sobel x direction and Sobel gradient magnitude thresholds were particularly effective, while Sobel y direction and, surprisingly, Sobel gradient direction were not effective.  Perhaps the Sobel gradient direction could have been improved by first applying [non-max supression](https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression).

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 98-138 and 228-272 in `video_gen.py`).  I tested out all possible combinations of `and` and `or` between the three input thresholds (`gradx`, `gradmag`, and `color`) and found that ((gradx & gradmag) | color) worked best.  Here's an example of my output for this step (note that the actual image returned is binary, color is just for visualization here):

![alt text][warpedPreprocessed]

and re-transformed onto the road:

![alt text][unwarpedPreprocessed]

Finally, I used a region of interest (seen in the Sliding windows figure below) to eliminate detections too far from where we expect them to be.  Note that in a fully self driving setting, this would be a bad idea because you still need to detect lanes even if you're straddling one!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I performed the perspective transform before applying the color threshold (reasons stated above).  I picked four "source" points on an image where the car was driving straight on a flat road, and used the `cv2.getPerspectiveTransform()` function to compute the transformation (and also its inverse) to four corners of a rectangle in a "bird's eye view" (code lines 210-212 in video_gen.py).  With this matrix, I then used the `cv2.warpPerspective()` function to warp the input image into the new view (code line 215 in video_gen.py).

Note that these points are hard coded and would need to change if the angle or height of the car with respect to the road changes (e.g., during bounces).  Here are the source and destination points.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 598,  446     | 320, 0        | 
| 682,  446     | 960, 0        |
| 1024, 673     | 960, 720      |
| 256,  673     | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warpCheck]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

##### 4.1 Sliding windows
Thresholded images (hopefully) containing lane line pixels are processed using functions in the `tracker` class inside `tracker.py`.  I used the sliding window convolution method outlined in the project description.  The function `tracker.find_window_centroids()` takes in an image and, while moving from the bottom to the top of the image, finds peaks in the convolution signal of the image and a small window (one peak in the left half and one peak in the right half).  Hyperparameters for the window include the width (50), the height (80, meaning there will be 720/80 vertical locations), and the margin, which determines how far to the left and right the window is allowed to slide away from its location in the next lowest layer (a higher margin allows for a curvier road, but also will potentially find more areas with false positives).

The centers of the peaks are then added to a list that records the locations of all peaks for all past frames. One thing I did differently from the code in the project description is I also kept track of the convolution signal and used it to filter the output where nothing was detected before fitting lane lines to the points.  The detected boxes for each frame along with the average boxes from the previous 15 frames are shown in green and red in the gif below.

![alt text][boxesGIF]

##### 4.2 Polynomial fitting
I tried several approaches to fitting polynomials to the detected window centers. Note that I fit with x along the image height dimension and y along the image width dimension because curved road lines might not be functions the other way around. Approaches I tested and their functional forms included:
* Fitting independent two degree polynomials for the left and right window centers from the last N frames
	- a_l * x * x + b_l * x + c_l and a_r * x * x + b_r * x + c_r
* Allowing the parameters of these polynomials to change with time (potentially achieving a better fit and possibly allowing to predict the lines in future frames based on past frames)
	- (a_l_i * t + a_l_j) * x * x + (b_l_i * t + b_l_j) * x + (c_l_i * t + c_l_j) and right line parameters
* Fitting the lane line data jointly by assuming polynomials for left and right lines must have same shape, but can be shifted left and right by an fitted parameter.
	- (a_i * t + a_j) * x * x + (b_i * t + b_j) * x + (c_i * t + c_j) + (line_sep_i) * line_ind
* Higher order polynomials in time for each of the parameters
	- (a_i * t * t * t + a_j * t * t + a_k * t + a_l) * x * x + (b_i * t * t * t + b_j * t * t + b_k * t + b_l) * x + (c_i * t * t * t + c_j * t * t + c_k * t + c_l) + (line_sep_i) * line_ind

The data matrix for the joint fit looks like this:

| Horizontal position (y) | Vertical position (x)| Lane indicator (right = 0) |
|:-----------------------:|:--------------------:|:--------------------------:| 
|             leftx       |       res_yvals      |              1             |
|            rightx       |       res_yvals      |              0             | 

I fit these functions to the data using my funciton `tracker.find_lane_fit_parameters()`, which ultimately calls the `scipy.optimize.curve_fit()`.  This function returns the parameters of the lane lines, which are used by the `tracker.get_line_fit_plot_points()` function to return a list of fitted lane points back to the call in `video_gen.py`.

Overall, I found that all these techniques give pretty much the same result, with the exception that allowing polynomial parameters to vary with time makes the lines somewhat more "wobbly".  Implementing a Kalman filter might be a better choice for next time.  Shown below is the video generated by jointly fitting the lane lines with line parameters allowed to vary with time

![alt text][trackedJointGIF]

and here is what it looks like fitting the lines independently.

![alt text][trackedSepGIF]

I used joint fits to try and improve detection robustness when one of the lanes was missing (and it does keep the top end of the left line from wandering off to the left as the car goes over the white section of pavement), but notice that the projection for the joint fitting method isn't as good when the car bounces up and down!  My assumption that the lines have the same shape parameters separated by a fixed distance not a good one if the perspective transform is incorrect.  You can see this in the perspective transformed gif below where normally parallel lines are no longer parallel during bounces.  Fitting the lines independently actually looks visually better for the "bouncing" periods, even if it's somewhat misleading about our actual knowledge about the world.  One next step could be to caculate a perturbation to the assumed projection based on how the difference distance between the two fitted lines changes as you move up and down the image.

![alt text][bouncingGIF]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in lines 347 through 348 in my code in `video_gen.py`.  To convert units from pixels to meters, I measured what I assumed was a 3 meter long road stripe (65 pixels) in the "bird's eye view", and also the distance between two lane lines (632 pixels), which I assumed was 3.7 meters.

The resulting radii of curvature (400-1000 meters) are the right order of magnitude for freeway curve radii.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I plotted the fitted lines and re-transforming back to the original perspective in lines 326 through 341 in my code in `video_gen.py`.  You can see the result in the tracked gifs above.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4), using joint fitting and allowing the parameters to vary linearly over the previous 15 frames which is also shown below as a gif:

![alt text][finalResult]

---

### Discussion

One potential improvement would be to use the convlution signal (i.e. number of pixels found in the window box) as weights for the fit.  This bias the fit towards using confident detections where the lane line is clear.

Another potential improvement would be to better estimate the perspective transform, or use the lane lines to determine a perterbation to the transform.  This would allow joint fitting of the left and right lane lines to do a better job.  It would also be interesting to track sections of the dashed lane to determine the vehicle's speed.

By far the biggest weakness of my pipeline is the color thresholding step.  Each new video has a different set of colors for the lines and pavement, which would be exaserbated by passing shadows, clouds, time of day, forests, etc.  My method of looking at the videos, measuring brightess of line regions, of road regions, and combining them into a threshold seemed to generate a new case for each "problem segment" in the video.  I have low confidence that these parameters would generalize to a new setting. For instance, my pipeline didn't to a great job on the challenge video (it just can't stay away from the edges on the cement barrier!)

![alt text][challengeResult]

Additional preprocessing techniques such as [adaptive thresholding](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.7883&rep=rep1&type=pdf) could be useful. Ultimately, why not let a deep neural network like [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) define the thresholds and how to combine them for us?