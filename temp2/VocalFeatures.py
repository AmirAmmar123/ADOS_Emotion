from ast import parse
import glob
from statistics import stdev
import numpy as np
import pandas as pd
import parselmouth
from Files import *
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statistics
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy
def two_pass_method(sound):
    # pitch = sound.to_pitch()
    pitch = call(sound, "To Pitch", 0.0, 50, 700) 
    # pitch_values = pitch.selected_array['frequency']
    # pitch_values[pitch_values==0] = np.nan
    # qs = pd.DataFrame(pitch_values).quantile([0.25,0.5,0.75])
    q1 = call(pitch,"Get quantile",0,0,0.25,"Hertz")
    q3 = call(pitch,"Get quantile",0,0,0.75,"Hertz")
    # print("Compare",qs.at[0.25,0],qs.at[0.75,0],q1,q3)
    # print(qs)
    return q1,q3

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    

    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
def runPCA(df):
    #Z-score the Jitter and Shimmer measurements
    features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    #y = df.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    #PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    return principalDf
def analyze_folder(folder_path : str,run_pca = False):
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    mean_intensity_list = []
    sd_intensity_list = []
    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob(folder_path+"/*.wav"):
        sound = parselmouth.Sound(wave_file)
        (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
        file_list.append(wave_file) # make an ID list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        hnr_list.append(hnr)
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
        intensity = sound.to_intensity()
    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list]),
                                columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer'])  #add these lists to pandas in the right order
    if run_pca:
        pcaData = runPCA(df)

        df = pd.concat([df, pcaData], axis=1)
    return df
def measureFormants(sound, f0min,f0max):
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)

    f1_sd = statistics.stdev(f1_list)
    f2_sd = statistics.stdev(f2_list)
    f3_sd = statistics.stdev(f3_list)
    f4_sd = statistics.stdev(f4_list)

    
    return [f1_mean, f2_mean, f3_mean, f4_mean], [f1_median, f2_median, f3_median, f4_median],[f1_sd,f2_sd,f3_sd,f4_sd]
def analyze_file(file_path : str,run_pca = False):
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []

    # Go through all the wave files in the folder and measure pitch
    sound = parselmouth.Sound(file_path)
    (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
    file_list.append(file_path) # make an ID list
    mean_F0_list.append(meanF0) # make a mean F0 list
    sd_F0_list.append(stdevF0) # make a sd F0 list
    hnr_list.append(hnr)
    localJitter_list.append(localJitter)
    localabsoluteJitter_list.append(localabsoluteJitter)
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    ddpJitter_list.append(ddpJitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    ddaShimmer_list.append(ddaShimmer)
    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list]),
                                columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer'])  #add these lists to pandas in the right order
    if run_pca:
        pcaData = runPCA(df)

        df = pd.concat([df, pcaData], axis=1)
    return df
def analyze_dataFrame(data,sample_freq,run_pca = False):
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    xmin_list = []
    xmax_list = []
    mean_Int = []
    std_Int = []
    # Go through all the wave files in the folder and measure pitch
    sound = parselmouth.Sound(data,sampling_frequency=sample_freq)
    q1,q3 = two_pass_method(sound)
    # (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 0.75*q1, 1.5*q3, "Hertz")
    print(q1,q3)
    (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 100, 500, "Hertz")

    means,meds,sds = measureFormants(sound,0.75*q1,1.5*q3)
    file_list.append("data") # make an ID list
    mean_F0_list.append(meanF0) # make a mean F0 list
    sd_F0_list.append(stdevF0) # make a sd F0 list
    hnr_list.append(hnr)
    localJitter_list.append(localJitter)
    localabsoluteJitter_list.append(localabsoluteJitter)
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    ddpJitter_list.append(ddpJitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    ddaShimmer_list.append(ddaShimmer)
    intensity = sound.to_intensity()
    meanInt= call(intensity, "Get standard deviation", 0 ,0)
    stdevInt= call(intensity, "Get mean", 0 ,0)
    mean_Int.append(meanInt)
    std_Int.append(stdevInt)
    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list,mean_Int,std_Int]),
                                columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer','meanIntensity','stdevIntensity'])  #add these lists to pandas in the right order
    df.at[0,'fdisp'] = (meds[0]-meds[3])/3
    df.at[0,'fdispSD'] = (sds[0]-sds[3])/3
    df.at[0,'avgFormant'] = (meds[0]+meds[1]+meds[2]+meds[3])/4
    df.at[0,'avgFormantSD'] = (sds[0]+sds[1]+sds[2]+sds[3])/4
    df.at[0,'mff'] = (meds[0]*meds[1]*meds[2]*meds[3])**0.25
    df.at[0,'mffSD'] = (sds[0]*sds[1]*sds[2]*sds[3])**0.25  
    df.at[0,'fitch_vtl'] = ((1 * (35000 / (4 * meds[0]))) +
                   (3 * (35000 / (4 * meds[1]))) + 
                   (5 * (35000 / (4 * meds[2]))) + 
                   (7 * (35000 / (4 * meds[3])))) / 4  
    df.at[0,'fitch_vtlSD'] = ((1 * (35000 / (4 * sds[0]))) +
                   (3 * (35000 / (4 * sds[1]))) + 
                   (5 * (35000 / (4 * sds[2]))) + 
                   (7 * (35000 / (4 * sds[3])))) / 4  
    
    xysum = (0.5 * meds[0]) + (1.5 * meds[1]) + (2.5 * meds[2]) + (3.5 * meds[3])
    xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    xysumSD = (0.5 * meds[0]) + (1.5 * meds[1]) + (2.5 * meds[2]) + (3.5 * meds[3])
    xsquaredsumSD = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    delta_f = xysum / xsquaredsum
    delta_fSD = xysumSD/xsquaredsumSD
    df.at[0,'delta_f'] = delta_f
    df.at[0,'vtl_delta_f'] = 35000 / (2 * delta_f)
    df.at[0,'delta_fSD'] = delta_fSD
    df.at[0,'vtl_delta_fSD'] = 35000 / (2 * delta_fSD)
    for formant in range(1,5):
        df.at[0,'formant'+str(formant)+'-mean']  = means[formant-1]
        df.at[0,'formant'+str(formant)+'-median']  = meds[formant-1]
        df.at[0,'formant'+str(formant)+'-SD']  = sds[formant-1]

    if run_pca:
        pcaData = runPCA(df)

        df = pd.concat([df, pcaData], axis=1)
    df.to_csv('test1.csv')
    return df
def get_pitch(data,sample_freq):
    sound = parselmouth.Sound(data,sampling_frequency=sample_freq)
    q1,q3 = two_pass_method(sound)
    pitch = call(sound, "To Pitch", 0.0, 0.75*q1, 1.5*q3,) #create a praat pitch object
    return pitch.selected_array['frequency']
def test():
    from scipy.io.wavfile import read
    rate,data= read(get_file())
    data = np.rot90(data)
    print(rate,data)
    print(analyze_dataFrame(data,rate))
    snd = parselmouth.Sound(data,rate)
    q1,q3=two_pass_method(snd)
    def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        plt.ylabel("fundamental frequency [Hz]")
    intensity= snd.to_intensity()
    pitch = snd.to_pitch()
    pitch2 = call(snd, "To Pitch", 0.0, 75, 600) 
    pitch3= call(snd, "To Pitch", 0.0, 0.75*q1, 1.5*q3,) #create a praat pitch object
    pitch1_values = pitch.selected_array['frequency']
    pitch2_values = pitch2.selected_array['frequency']
    pitch3_values = pitch3.selected_array['frequency']
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz") # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
    stdevInt= call(intensity, "Get standard deviation", 0 ,0)
    means,meds,sds = measureFormants(snd,0.75*q1,1.5*q3)
    # dispersion = (meds[1]-meds[4])/3
    print(means,meds,sds)
    print("pitch mean 1: ",np.mean(pitch1_values))
    print("pitch mean 2: ",np.mean(pitch2_values))
    print("pitch mean 1: ",np.mean(pitch1_values[pitch1_values!=0]))
    print("pitch mean 2: ",np.mean(pitch2_values[pitch2_values!=0]))
    print("pitch mean 3: ",np.mean(pitch3_values[pitch3_values!=0]))
    print("pitch avg 1:",np.average(pitch.selected_array['frequency']))
    print("pitch avg 2:",np.average(pitch2.selected_array['frequency']))
    
    print("pitch meanF0: ",meanF0)
    print("pitch avg 1:",np.std(pitch.selected_array['frequency']))
    print("pitch avg 2:",np.std(pitch2.selected_array['frequency']))
    print("pitch stdevF0: ",stdevF0)
    print(stdevInt)
    print("intensity:",np.std(intensity.values))
    # print("pitch avg 1:",pitch.fmax)
    # print("pitch avg 2:",pitch2.fmax)
    print("intensity:",meanF0)  
    print("intensity:",np.mean(intensity.values))
    print("intensity:",intensity.get_average())
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")draw_pitch(pitch)
    draw_pitch(snd.to_pitch())
    plt.xlim([snd.xmin, snd.xmax])
    plt.show()
    # pitch = snd.to_pitch()
    # pitch_values = pitch.selected_array['frequency']
    # pitch_values[pitch_values==0] = np.nan
    # print(pd.DataFrame(pitch_values).quantile([0.25,0.5,0.75]))
    # print(two_pass_method(snd))

# test()