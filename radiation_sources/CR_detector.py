
import cv2
import time
import numpy as np
import logging

"""
    Cosmic Ray (particle) detection script developed in previous research project.
"""


# Parameters
fps = 40
threshold = 30
padding = 10
frame_count = 0
count = 0

# Resolution
frame_width = 1920
frame_height = 1080

# Logging setup
logger = logging.getLogger('particle_detection')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/particle_detection.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Camera properties
cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

logger.info("Starting detection...")
print("Detecting")
logger.info(f'fps : {fps}, threshold : {threshold}, resolution : {frame_width}x{frame_height}')
print(f'fps : {fps}, threshold : {threshold}, resolution : {frame_width}x{frame_height}')
print("Press ^C to stop detection")

try:
    while True:
        frame_count += 1
        # Read current frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera not working")
            print("Camera not working")
            continue
        # Pre-processing: grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Morphological Operation: Opening to reduce small noise
        #kernel_size = 3
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        #opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        # Contour extraction


        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours :
            # Process each contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x -= padding
                y -= padding
                w += 2*padding
                h += 2*padding
                cosmic_ray = frame[y:y+h, x:x+w]
                if cosmic_ray is None or len(cosmic_ray) == 0:
                    continue
                # Ensure the slice is in color
                #if cosmic_ray.ndim == 2:
                #   cosmic_ray = cv2.cvtColor(cosmic_ray, cv2.COLOR_GRAY2BGR)
                timestamp = time.strftime('%d.%m_%H.%M.%S')
                # Save images
                cv2.imwrite(f'detections/detection_{count}_{timestamp}.png', cosmic_ray)
                cv2.imwrite(f'frames/frame_{count}_{timestamp}.png', frame)
                logger.info(f'Saved image detection_{count}_{timestamp}.png')
                print((f'Saved image detection_{count}_{timestamp}.png'))
                count += 1
            
except KeyboardInterrupt:
    logger.info('Detection stopped by user')
except Exception as e:
    logger.error(f'Error occurred: {e}')
finally:
    cap.release()
    logger.info('Detection finished')