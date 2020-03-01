import cv2#importing opencv
import sys
loc=sys.argv[1]#image name as command-line argument
cascloc=sys.argv[2]#cascade name as command-line argument
faceCascade=cv2.CascadeClassifier(cascloc)#creating the haar cascade
image=cv2.imread(loc)#reading the image
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#converting it to grayscale
#main code for face detection
faces=faceCascade.detectMultiScale(#General function that detects objects
    gray,#grayscale image
    scaleFactor=1.1,#compensates the size of the faces
    minNeighbors=5,#how many objects are detected near the current one
    minSize=(30, 30)#gives the size of each window
)
print("Found {0} faces!".format(len(faces)))
for (x,y,w,h)in faces:#Draws a rectangle around the faces
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)#x and y are location while w- width and h- height
cv2.imshow("Faces -->", image)
cv2.waitKey(0)
# run this code on cmd use -->
# python <name_of_your_python_file> <image_file>.png haarcascade_frontalface_default.xml
#haarcascade_frontalface_default.xml is a haar cascade designed by OpenCV to detect the frontal face.
