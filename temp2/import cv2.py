import cv2
from Files import get_file, get_folder
from os import mkdir
vidcap = cv2.VideoCapture(get_file("Select video file to split to images"))
folder = get_folder("select output folder")

success,image = vidcap.read()
count = 0
mkdir(folder+"/sub")
sub = folder+"/sub"
timer = 0
max  = 15
while success:
  success,image = vidcap.read()
  # cv2.imwrite(folder+"/"+"frame%d.jpg" % count, image)     # save frame as JPEG file
  if(count%max==0):
    cv2.imwrite(sub+"/"+"frame%d.jpg" % count, image)
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1