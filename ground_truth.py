import cv2
import pandas as pd
import matplotlib.pyplot as plt

tracker = cv2.TrackerCSRT_create()

tracking_box = [] # track x and y coordinates for ground truth comparsion

# Read video
video = cv2.VideoCapture("BlueBounce.mp4")
video.set(1,9)
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
bbox = (35, 338, 130, 130)

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

#print(bbox)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)
    print(bbox[0:2])
    # track progress
    tracking_box.append(bbox)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
            # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, "CSRT" + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)


    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

csrt_trajectory = tracking_box[:-20] # keep it at 60 frames
csrtDF = pd.DataFrame(csrt_trajectory, columns = ['x','y','drop1','drop2'])
csrtDF.drop(columns = ['drop1','drop2'], axis = 0, inplace = True)

#csrtDF.to_csv('csrt_trajectory.csv')
print(len(csrt_trajectory))# 60 frames 
