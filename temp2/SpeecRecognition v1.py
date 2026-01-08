# based on https://realpython.com/python-speech-recognition/#how-speech-recognition-works-an-overview
import speech_recognition  as sr
from Files import *
from bidi.algorithm import get_display
r = sr.Recognizer()
myaf = sr.AudioFile(get_file())
with myaf as src:
    audio = r.record(src)
print(type(audio))
print(get_display(r.recognize_google(audio,language="he")))
