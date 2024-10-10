import cv2
from sklearn.cluster import KMeans
import numpy as np


image_path = 'red.png'  
 # Loading the cone image
image = cv2.imread(image_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Image turned to HSV color space for better color detection

# HSV color range is defined for red cones
# Red has two hue ranges in HSV, thus lower threshold used for bright reds
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# Noise reduction and isolation of cones.
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Contour detection from mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Center points of each detected cone for further analysis is stored
cone_points = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  # Bounding box for contour
    x_center = x + w // 2  # X-coordinate of the cone's center
    y_center = y + h // 2  # Y-coordinate of the cone's center
    cone_points.append([x_center, y_center])  # Append the cone center coordinates

# Convert list to numpy array for clustering
cone_points = np.array(cone_points)

# K-Means clustering used to separate cones into L and R side
kmeans = KMeans(n_clusters=2, random_state=0).fit(cone_points)
labels = kmeans.labels_

# Separate points into two clusters based on the K-Means labels
L_cones = cone_points[labels == 0]
R_cones = cone_points[labels == 1]

# Visualizing the detected cones by making circles around each
for point in L_cones:
    cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)  # Green

for point in R_cones:
    cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)  # Blue 

# Putting a straight line to the left and right cones
if len(L_cones) > 1:  # Ensure we have enough points to fit a line
    [vx, vy, x, y] = cv2.fitLine(L_cones, cv2.DIST_L2, 0, 0.01, 0.01)
    L_LineStart = (int(x.item() - vx.item() * 1000), int(y.item() - vy.item() * 1000))
    L_LineEnd = (int(x.item() + vx.item() * 1000), int(y.item() + vy.item() * 1000))
    cv2.line(image, L_LineStart, L_LineEnd, (0, 0, 255), 2)  # Red line

if len(R_cones) > 1:  # Ensure we have enough points to fit a line
    [vx, vy, x, y] = cv2.fitLine(R_cones, cv2.DIST_L2, 0, 0.01, 0.01)
    R_LineStart = (int(x.item() - vx.item() * 1000), int(y.item() - vy.item() * 1000))
    R_LineEnd = (int(x.item() + vx.item() * 1000), int(y.item() + vy.item() * 1000))
    cv2.line(image, R_LineStart, R_LineEnd, (0, 0, 255), 2)  # Red line


# Saving final image
output_path = 'answer.png'  
cv2.imwrite(output_path, image)

# Final message indicating the saved file
print("Boundary image saved")
