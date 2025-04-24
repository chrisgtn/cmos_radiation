import cv2
import time
import numpy as np
import logging
import os

""" 
    Proton detection script (adjusted from the Cosmic Ray detector script) aimed for the proton beam experiment. 
"""

# Parameters
fps = 40
threshold = 60
frame_count = 0

# Resolution
frame_width = 1920
frame_height = 1080


# Generate a unique run ID and directory for this run
run_id = time.strftime('%Y%m%d_%H%M%S')
run_directory = os.path.join('runs', run_id)

if not os.path.exists(run_directory):
    os.makedirs(run_directory)
    

logs_directory = 'logs'
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)
    
# Logging setup
log_file_name = f'proton_detection_{run_id}.log'  
log_file_path = os.path.join(logs_directory, log_file_name) 

logger = logging.getLogger('proton_detection')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Camera properties
cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    
logger.info("Proton detection")
print(f"Detecting run_ID : {run_id}")
logger.info(f'fps : {fps}, threshold : {threshold}, resolution : {frame_width}x{frame_height}')
print(f'fps : {fps}, threshold : {threshold}, resolution : {frame_width}x{frame_height}')
print("Press ^C to stop detection")



# Capture and save the initial frame
ret, initial_frame = cap.read()

if ret:
    # Initial frame
    initial_frame_path = os.path.join(run_directory, 'initial_frame.png')
    cv2.imwrite(initial_frame_path, initial_frame)
    logger.info('Saved initial frame')
else:
    logger.error("Failed to capture the initial frame.")
    cap.release()
    #exit()


try:
    while True:
        
        # Read current frame
        ret, frame = cap.read()

        if not ret:
            logger.error("Camera not working")
            print("Camera not working")
            continue

        # grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        

        # Gaussian Blurring to smooth the image
        #kernel_size = (5, 5)
        #blurred_gray = cv2.GaussianBlur(gray, kernel_size, 0)
        
        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if contours :
                frame_count += 1
                timestamp = time.strftime('%d.%m_%H.%M.%S')
                frame_filename = f'frame_{frame_count}_{timestamp}.png'
                frame_path = os.path.join(run_directory, frame_filename) 
                    
                # Save frames
                cv2.imwrite(frame_path, frame) 
                logger.info(f'Saved image {frame_path}')
                print(f'Saved image {frame_path}')

           
except KeyboardInterrupt:
    logger.info('Detection stopped by user')
except Exception as e:
    logger.error(f'Error occurred: {e}')
finally:
    cap.release()
    logger.info('Detection finished')

