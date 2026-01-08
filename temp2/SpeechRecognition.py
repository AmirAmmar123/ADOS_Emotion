import speech_recognition as sr
import numpy as np
from Files import *
import time
r = sr.Recognizer()
first= -1
last = -1
audio_file = sr.AudioFile(get_file())
for t in np.arange (0, 10, 0.1):
    with audio_file as source:
        print(t)
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            result = (r.recognize_google(r.record(source,offset=t,duration=2),language='iw-IL'))
            if('שיניים' in result):
                if first==-1:
                    first = t
                print(result[::-1],' Now!')
                if first>-1:
                    last = t
            # print(type(result))
            # time.sleep(1)
        except:
            continue
print(first,last)
    # print(r.recognize_google(r.record(source),language = 'hebrew'))