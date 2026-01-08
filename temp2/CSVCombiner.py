from Files import *
import pandas as pd
from pathlib import Path
files_to_combine = get_files()
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

for i,file in enumerate(files_to_combine):
    for j,otherFile in enumerate(files_to_combine):
        filePath = Path(file)
        otherFilePath = Path(otherFile)
        fileName = filePath.name
        otherFileName = otherFilePath.name
        if(fileName == otherFileName):
            continue
        fileName = fileName[4:]
        otherFileName = otherFileName[4:]
        # print(fileName,otherFileName)
        if fileName.replace('1','2') == otherFileName:
            firstPart = pd.read_csv(filePath)
            secondPart = pd.read_csv(otherFilePath)
            secondPart['frame'] = secondPart['frame']+firstPart['frame'].max()
            timestamp_str = ' timestamp'
            if ' timestamp' not in list(secondPart.columns):
                for col in secondPart.columns:
                    if 'timestamp' in col:
                        timestamp_str = col
            secondPart[timestamp_str] = secondPart[timestamp_str]+firstPart[timestamp_str].max()  + (firstPart.at[1,timestamp_str] -firstPart.at[0,timestamp_str])
            # print(str(firstPart))
            # print(str(secondPart))
            combined = pd.concat([firstPart,secondPart],ignore_index=True)
            # print(str(combined))
            new_path = (rreplace(filePath.__str__(),'1','',1))
            # combined.re/
            combined.to_csv(new_path,index=False)

            # exit(1)

