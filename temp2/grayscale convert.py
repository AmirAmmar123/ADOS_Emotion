from PIL import Image
from Files import get_file, get_folder,get_files
folder = get_files("Select files of images to convert")
ofolder = get_folder("Select folder of ouput")
count=0
for img in folder:
    name = img.split("/")[-1]
    img = Image.open(img).convert('L')
    img.save(ofolder+"/"+name)