import numpy as np
import cv2 as opencv
import argparse
from collections import deque

# Capture video from webcam
video_capture = opencv.VideoCapture(0)
# Show a line on the image tracing the centroid.
points = deque(maxlen=64)

# Set up colors to detect.
Lower_green = np.array([110, 50, 50])
Upper_green = np.array([130, 255, 255])

while True:
    # capture video from feed.
    ret, image_from_video = video_capture.read()

    # convert rgb to hsv.
    hsv_image = opencv.cvtColor(image_from_video, opencv.COLOR_BGR2HSV)

    # set a detection kernel.
    kernel = np.ones((5, 5), np.uint8)

    # Reduces the matrix to a mask which lies in our color range.
    mask = opencv.inRange(hsv_image, Lower_green, Upper_green)

    # Erode the resultant mask to remove the small aberrations
    mask = opencv.erode(mask, kernel, iterations=2)

    # MORPH_OPEN does an erosion followed by dilation. This helps remove white noise from image.
    mask = opencv.morphologyEx(mask, opencv.MORPH_OPEN, kernel)

    # mask=opencv.morphologyEx(mask,opencv.MORPH_CLOSE,kernel)
    mask = opencv.dilate(mask, kernel, iterations=1)

    # The bitwise_and joins the two images and mask shows the area of interest
    colored_mask = opencv.bitwise_and(
        image_from_video, image_from_video, mask=mask)

    # Find contours from the mask.
    contours, heir = opencv.findContours(
        mask.copy(), opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)[-2:]
    centroid = None

    # If a contour is detected
    if len(contours) > 0:
        # Get contour of maximum size
        largest_contour = max(contours, key=opencv.contourArea)

        # Get a bounding circle.
        ((x, y), radius) = opencv.minEnclosingCircle(largest_contour)

        # Calculate the moments to find the centroid of the largest contour.
        M = opencv.moments(largest_contour)
        # Check https://en.wikipedia.org/wiki/Image_moment for reference.
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # IF the bounding circle is large enough.
        if radius > 5:
            # draw that circle on the image.
            opencv.circle(image_from_video, (int(x), int(y)),
                          int(radius), (0, 255, 255), 2)
            # Also draw the centroid on the image
            opencv.circle(image_from_video, centroid, 5, (0, 0, 255), -1)

    # Append the new centroid to the trace line.
    points.appendleft(centroid)

    # Draw the line.
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        # thickness dies out for older points due to 'i' factor.
        thick = 2  # int(np.sqrt(len(points) / float(i + 1)) * 2.5)
        opencv.line(image_from_video,
                    points[i-1], points[i], (0, 0, 225), thick)

    # Show the images
    opencv.imshow("Image captured from video with trace line.",
                  opencv.flip(image_from_video, 1))
    opencv.imshow("Mask of the tracing point.", mask)
    opencv.imshow(
        "Snippet from the original image being traced.", colored_mask)

    # Wait for a key press event. In this case, wait for at least 30 ms for 'Space' key.
    k = opencv.waitKey(30) & 0xFF
    if k == 32:
        break

# cleanup the camera and close any open windows
video_capture.release()
opencv.destroyAllWindows()
