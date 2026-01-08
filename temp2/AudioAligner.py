from facesync.facesync import facesync
from Files import *
# change file name to include the full
video_files = [get_file()]
target_audio = get_file()
# Intialize facesync class
fs = facesync(video_files=video_files,target_audio=target_audio)
# Extracts audio from sample1.MP4
fs.extract_audio()
# Find offset by correlation
fs.find_offset_corr(search_start=14,search_end=16)
print(fs.offsets)
# Find offset by fast fourier transform
# fs.find_offset_fft()
# print(fs.offsets)