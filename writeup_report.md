##Vehicle Detection And Tracking

[//]: # (Image References)
[image1]:  ./output_images/1_car_noncar.png
[image2]:  ./output_images/2_hog_car_noncar.png
[image3]:  ./output_images/3_detection.png
[image4]:  ./output_images/4_detection.png
[image5]:  ./output_images/5_detection.png
[image6]:  ./output_images/heat_1.png
[image7]:  ./output_images/heat_2.png
[image8]:  ./output_images/heat_3.png
[image9]:  ./output_images/heat_4.png
[image10]: ./output_images/heat_5.png
[image11]: ./output_images/heat_6.png
[image12]: ./output_images/heat_cars.png
[image13]: ./output_images/heat_boxes.png
[video1]:  ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I am using the `hog()` function from the package `skimage.feature` to extract the HOG features from the training images. The  hog feature extraction code can be found in the function named `get_hog_features()` in lines 6 through 24 in the file `utils.py`. The parameters for the hog feature extraction are set in lines 103 through 106 in the file named `main.py`.

To identify the correct parameters for HOG I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Then I trained the dataset with various combinations of parameters to the `skimage.hog()`. I found the following winning combination of the hog parameters which resulted in a test accuracy of ~99%.
`
orient = 10 
pix_per_cell = 8
cell_per_block = 2 
hog_channel = 'ALL'
`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The dataset provided contains 17,760 images of 64x64 pixels, of with 8792 samples labeled as car and 8968 sampled labeled as non-car. I used `LinearSVC` from the `sklearn.svm` package for training the dataset. I used the following combination of parameters to extract features from the dataset for training. The code the configuration of the parameters can be found in lines 101 through 113 in the file `main.py`.

`
color_space = 'YCrCb'
orient = 10
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 16
`
Lines 47 through 95 in the file `utils.py` contains the functions for feature extraction. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did multiple test runs with different overlap and scale settings to find a suitable configuration. I settled with the following configuration for the sliding window:
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                    xy_window=(144, 144), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[430, 550],
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[460, 580],
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
					
The `slide_window()` function is defined in lines 101 through 140 in the file `utils.py`. The function is called with proper parameters in lines 164 through 171 and lines 192 through 198 in the file `main.py`.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

