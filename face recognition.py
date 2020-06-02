
from imutils.video import VideoStream
from twilio.rest import Client
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import os
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())


print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()
f=True
# loop over frames from the video file stream
while True:
 
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        time.sleep(1)
        name = "Unknown"
        # check to see if we have found a match
        
        if True in matches:
            f=True
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        # update the list of names
        names.append(name)

          
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
    # display the image to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()
    
    for encoding in encodings:
    
        if not True in matches:
            name="Unknown"
            if f==True:
                print("Alerts!unknown person")

# Start saving video to the unknown person
                f_id=1
                FILE_OUTPUT = 'output.avi'
                start_time=time.time()

                while os.path.isfile(FILE_OUTPUT):
                    f_id += 1
                    FILE_OUTPUT="output.%s.1.avi" % f_id
                    
# Checks and deletes the output file
# You cant have a existing file or it will through an error
#if os.path.isfile(FILE_OUTPUT):
 #   os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
                cap = cv2.VideoCapture(0)
                currentFrame = 0

# Get current width of frame
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


# Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'X264')
                out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))

                while(True):
                #while(cap.isOpened()):
                    
                    
    # Capture frame-by-frame
                    
                        
        # Saves for video
                    out.write(frame)

        # Display the resulting frame
                        
                    end_time=time.time()
                    elapsed = end_time - start_time
                    if elapsed > 10:
                        break
             
    # To stop duplicate images
                        currentFrame += 1

# When everything done, release the capture
                    cap.release()
                    out.release()
                print("Video Saved.")

#Sendinf the saved video of the unknown person via Email


                

                

#Send SMS
               TWILIO_SID = "AC5dd4f3412ee73c41b76c12be1802d3e1"
               TWILIO_AUTH = "a692cb465ae735cf88ed59ded6de3fa6"
               client = Client(TWILIO_SID, TWILIO_AUTH)
               TO = "phone number 1"
               FROM "phone number 2"
client.messages.create(to=TO, from_=FROM, body="Alert! unknown person is detected"
                f=False
  
# stop the timer and display FPS information
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
