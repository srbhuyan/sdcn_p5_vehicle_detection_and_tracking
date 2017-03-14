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
[image14]: ./output_images/windows.png
[video1]:  ./project_video_out.mp4

###Histogram of Oriented Gradients (HOG)

I am using the `hog()` function from the package `skimage.feature` to extract the HOG features from the training images. The  hog feature extraction code can be found in the function named `get_hog_features()` in lines 7 through 24 in the file `utils.py`. The parameters for the hog feature extraction are set in lines 103 through 106 in the file named `main.py`.

To identify the correct parameters for HOG I started by reading in and analyzing random samples from the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` (image on left) and `non-vehicle` (image on right) classes.

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####HOG parameters

To get to an optimum set of HOG parameters to use for feature extraction, I trained our dataset with various values of the `orient`, `pix_per_call` and `hog_channel` parameters. I found the following winning combination of HOG parameters which resulted in a test accuracy of ~99% (in combination with spatial and color histogram features).

    orient = 10
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

####Training

The dataset provided contains 17,760 images of 64x64 pixels, of with 8792 samples labeled as car and 8968 sampled labeled as non-car. We used spatial, color channel histogram and hog features for training the classifer.

####Spatial Features
For the spatial features the images were resized to (16,16) and flattened.

     features = cv2.resize(img, size).ravel()
       
####Histogram Features
Color histogram information from each channel was extracted by using 16 bins and a range of (0, 255).

    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

NOTE: HOG feature extraction if described in the section above named 'HOG parameters'.

Lines 47 through 95 in the file `utils.py` contains the functions for feature extraction. 

###Sliding Window Search

####Sliding Window
I use a sliding window approach to detect vehicles in the video frame. I did multiple test runs with different overlap and scale settings to find a suitable configuration. I settled with the following configuration for the sliding window:
    
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                    xy_window=(80, 80), xy_overlap=(0.5, 0.5))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 600],
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700],
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))

Following is a visual of the windows used.

![alt text][image14]

The `slide_window()` function is defined in lines 101 through 140 in the file `utils.py`. The function is called with proper parameters in lines 164 through 171 and lines 192 through 198 in the file `main.py`.

####Pipeline Performance

To get an optimum result I used the feature extraction parameters described in the 'Training' section above. I used the YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images of the result I achieved:

![alt text][image3]
![alt text][image4]
![alt text][image5]
---

### Video Implementation

####Video Result
Here's a [link to my video result](./project_video_out.mp4)


####Filtering False Positives and Combining Overlapping Bounding Boxes

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I am tracking 10 frames (line 282 in `main.py`) of the video for overlapping boxes and to build the heatmap. I am using a threshold of 5 (line 283 in `main.py`) to filter false positives. The implementation can be found in the functions defined in lines 230 through 277 in file `main.py`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]



---

###Discussion

####Tracking Overlap: 
My current implementation does not clearly separate the bounding boxes When two vehicles are close to each other. This can be improved by implmenting a mechanism to track the moving centroid of the vehicles, so that the system can predict where a vehicle might appear in subsequent frames.

####Car vs Non-Car Objects: 
This model works well on the freeways, but it might fail if there are pedestrians/road works etc. on the road. To improve and generalize the model further to correctly detect cars from non-car objects we will need to train with a much lager dataset.

