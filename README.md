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

[detectedCorners]: ./output_images/corners_found4.jpg "Detected corners"
[undistortedCorners]: ./output_images/undistorted_corners_found4.jpg "Original image and undistorted, perspective transformed result."
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. My project includes the following files:
* [camera_calibration.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/XXX.py) - a Python script to generate camera matrix and distortion coefficients from checkerboard images
* [image_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/XXX.py) - a Python script to run the tracking pipeline on sample images
* [video_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/XXX.py) - a Python script to run the tracking pipeline on sample videos
* [tracker.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/XXX.py) - a Python class that handles lane finding and fitting
* [project_video_tracked.mp4](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/XXX.mp4) - the output of the lane finding pipeline on the sample video.
* [README.md, this document](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/README.md) - a writeup report summarizing the results

### Camera Calibration

#### 1. OpenCV makes it a sinch to calculate a camera matrix and distortion parameters from a list of checkerboard images (e.g., "Original Image" in the figure below).  The code for this camera calibration is contained in camera_calibration.py.

The first step uses the `cv2.findChessboardCorners()` function to detect strong corners in an image and fit to these points a checkerboard of a specified number of squares. The image below shows an example of detected corners:

![alt text][detectedCorners]

Next, I used `cv2.calibrateCamera()` function to fit a 3D set of checkerboard points to the detected image points (code lines 51-69 in camera_calibration.py).  The function takes in a set of 3D locations of the corner points in local checkerboard coordinates (i.e. z = 0 for all points), a set of 2D pixel locations of the corresponding corner points in image coordinates and determines the rotation and translation vector of the checkerboard relative to the camera so that the reprojection error for the estimated 3D world coordinates of the checkerboard points is minimized.  We save the camera matrix and undistortion coefficients for later use (code lines 65-69 in camera_calibration.py)

Finally, I used the calculated camera matrix (which contains the focal lengths and centers of projection) and undistortion coefficients to correct for lense distortion (code lines 77-78 in camera_calibration.py) and compute a perspective transformation (using the `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions inside my `corners_unwarp()` function) that shows the checkerboard from straight ahead (code lines 73-113 in camera_calibration.py).  The undistorted, perspective transformed result is shown below:

![alt text][undistortedCorners]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
