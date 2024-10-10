![answer](https://github.com/user-attachments/assets/dd127503-0800-45e6-a9c8-f58b64b94905)

Cone detection using openCV and python

Overview:
This project uses OpenCV to detect cones in an image and draw boundary lines for a path between them. The cones are color-detected, clustered, and used to fit red lines representing the boundaries of the path.

Files:

main.py: The Python script for cone detection and boundary drawing.
answer.png: The output image with boundary lines.

Methodology: 

Load Image: The input image (red.png) is loaded.
Convert to HSV: The image is converted to HSV color space for easier color detection.
Thresholding: A mask is applied to isolate red cones using HSV ranges.
Find Contours: Contours are detected, and the center of each cone is calculated.
Cluster Cones: KMeans clustering groups cones into two sets (left and right).
Draw Circles: Red circles are drawn on each detected cone.
Fit Lines: Red lines are fitted through both the left and right cones.
Save Output: The processed image is saved as answer.png.

What I Tried:

I tried using K-Means Clustering to split cones into a left and right group. It took me a while to learn how to use this, as it was something to do with machine learning and completely new to me, but i think i was able to successfully implement it. I think there are easier ways to do this task, and K-means clustering wasnt actually needed. Im still not sure why the red line wasnt able to follow the cones though?
Morphological Operations: Applied for noise reduction but isnt essential for this image.

Libraries Used:

OpenCV: For image processing.
NumPy: For handling arrays and coordinates.
KMeans: From sklearn for clustering cone points.

How to Run:

Install these libraries:

pip install opencv-python numpy scikit-learn

Run the script:

python main.py

The output image will be saved as answer.png.
