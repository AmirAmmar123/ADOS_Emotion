import scipy.io.wavfile
import scipy.signal
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from skimage.morphology import binary_opening
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, os.path




def StandardizeScaler(feat):
    feat_scaled = (feat - np.mean(feat))/np.std(feat)
    return feat_scaled

def NormalizeScaler(feat):
    feat_scaled = (feat - np.min(feat))/(np.max(feat)-np.min(feat))
    return feat_scaled


## filter selected AUs in dataframe by success
def PrepareSignal(df,au_list):

    au_df = df[au_list]

    # clean for success =1

    for au in au_list:
        
        for idx, row in df.iterrows():

            if df['success'].iloc[idx] == 0:
                au_df.iloc[idx] = 0

    return df



## remove frames in which sub is speaking
def RemoveSubSpeakingFrames(df,au_list):

    au_df = df[au_list]

    # clean for success =1

    for au in au_list:
        
        for idx, row in df.iterrows():

            if df['sub_talking'].iloc[idx] == 0:
                au_df.iloc[idx] = 0

    return df   


def calculate_frame_rate(df):
    df = df.sort_values(by='timestamp')
    time_diff = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    total_frames = len(df)
    return total_frames / time_diff if time_diff > 0 else 30


## smooth the AU signal with scipy
def SmoothSignal(df, au, lowpass, fr ):
    times = np.arange(len(df))/fr
    # apply a 3-pole lowpass filter at 0.02x Nyquist frequency
    b, a = scipy.signal.butter(N=3, Wn=0.02,btype='lowpass', analog=False)
    au_smooth = scipy.signal.filtfilt(b, a, df[au], method='pad',padlen=200)
    au_smooth = np.where(au_smooth<0, 0, au_smooth)
    return au_smooth




## find high and low peaks with scipy
def FindHLPeaks(df, au_smooth, height, prominence):
    ###### FIND HIGH PEAKS #########
    value_hp = np.array(au_smooth)
    peak_hp, properties_hp = find_peaks(value_hp, height=height,distance = 30,width=30,prominence = prominence)
    if len(peak_hp) == 0:
        peak_hp = []
    ###### FIND LOW PEAKS #####
    value_lp = np.array(au_smooth)
    peak_lp, properties_lp  = find_peaks(-value_lp, prominence=prominence)

    if len(peak_lp) == 0:

        peak_hp = []  

    #add a closing peak_lp at the end of signal
    #end = df.tail(1).index.item()
    #peak_lp = np.append(peak_lp, end)

    return peak_hp, properties_hp, peak_lp, properties_lp



## get features from smoothed signal and peaks for SINGLE AU
def ComputeSingleAUPeaksFeatures( df, au_df , peak_hp, peak_lp, height, fr):

    times = np.arange(len(df))/fr

    ##### COMPUTE PEAKS FETURES ##########
    peaks_int = []
    peaks_prom = []
    peaks_area = []
    peaks_dur = []
    peaks_lp = []

    au_bin = pd.DataFrame(np.array([0] * len(au_df)))

    # check if any range between lower peaks contains higher peaks

    if len(peak_hp) != 0:
        for start,stop in zip(peak_lp, peak_lp[1:]):


        
            for elem in peak_hp:
                if start < elem < stop:

                    if float((stop - start)/fr) > 0.5:

                        current_peak = au_df.iloc[start:stop]
                        current_peak = current_peak.loc[(current_peak > height).any(axis=1)]

                        # compute features
                        peak_prom = float(np.max(current_peak))
                        peak_int = float(np.mean(current_peak))
                        peak_area = float(np.sum(current_peak))
                        peak_dur = float(len(current_peak)/fr)
                        au_bin.iloc[start:stop] = np.where(au_df.iloc[start:stop] > 0, 1, 0)


                        # save features
                        peaks_int.append(peak_int)
                        peaks_prom.append(peak_prom)
                        peaks_area.append(peak_area)
                        peaks_dur.append(peak_dur) 
                        peaks_lp.append((start, stop))  

    elif len(peak_hp) == 0:

        # save features
        peaks_int.append(np.nan)
        peaks_prom.append(np.nan)
        peaks_area.append(np.nan)
        peaks_dur.append(np.nan) 
        peaks_lp.append(np.nan)


            
    
    peaks_dict = dict = {'int_avg_combo': peaks_int, 'prominence_combo': peaks_prom, 'area_combo': peaks_area,'duration_combo': peaks_dur, 'peak_lp_combo':peaks_lp} 
    peaks_df = pd.DataFrame(peaks_dict)

    return peaks_df, au_bin




## get features from smoothed signal and peaks for COMBO AU
def ComputeComboAUPeaksFeatures( df, aus_dict, peak_hp_dict, peak_lp_dict, height, fr):
    times = np.arange(len(df))/fr
    

    #SIMPLE
    peaks_int_1_simple = []
    peaks_prom_1_simple = []
    peaks_area_1_simple = []
    peaks_dur_1_simple = []
    peaks_lp_1_simple = []

    peaks_int_2_simple = []
    peaks_prom_2_simple = []
    peaks_area_2_simple = []
    peaks_dur_2_simple = []
    peaks_lp_2_simple = []

    peaks_int_3_simple = []
    peaks_prom_3_simple = []
    peaks_area_3_simple = []
    peaks_dur_3_simple = []
    peaks_lp_3_simple = []

    #COMBO
    peaks_int_1_combo = []
    peaks_prom_1_combo = []
    peaks_area_1_combo = []
    peaks_dur_1_combo = []
    peaks_lp_1_combo = []

    peaks_int_2_combo = []
    peaks_prom_2_combo = []
    peaks_area_2_combo = []
    peaks_dur_2_combo = []
    peaks_lp_2_combo = []

    peaks_int_3_combo = []
    peaks_prom_3_combo = []
    peaks_area_3_combo = []
    peaks_dur_3_combo = []
    peaks_lp_3_combo = []

    peaks_int_4_combo = []
    peaks_prom_4_combo = []
    peaks_area_4_combo = []
    peaks_dur_4_combo = []
    peaks_lp_4_combo = []

    au_bin_1_simple = pd.DataFrame(np.array([0] * len(df)))    
    au_bin_1_combo = pd.DataFrame(np.array([0] * len(df)))


    print('Number of given AUs: ',len(list(aus_dict.keys())))

    if len(list(aus_dict.keys()))==2:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]


        # SIMPLE

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]

                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]

                                # save au1 features for combo
                                peak_prom_1_combo = float(np.max(current_peak_1))
                                peak_int_1_combo = float(np.mean(current_peak_1))
                                peak_area_1_combo = float(np.sum(current_peak_1))
                                peak_dur_1_combo = float(len(current_peak_1)/fr)

                                # save au2 features for combo
                                peak_prom_2_combo = float(np.max(current_peak_2))
                                peak_int_2_combo = float(np.mean(current_peak_2))
                                peak_area_2_combo = float(np.sum(current_peak_2))
                                peak_dur_2_combo = float(len(current_peak_2)/fr)

                                # add features to list
                                peaks_int_1_combo.append(peak_int_1_combo)
                                peaks_prom_1_combo.append(peak_prom_1_combo)
                                peaks_area_1_combo.append(peak_area_1_combo)
                                peaks_dur_1_combo.append(peak_dur_1_combo) 
                                peaks_int_2_combo.append(peak_int_2_combo)
                                peaks_prom_2_combo.append(peak_prom_2_combo)
                                peaks_area_2_combo.append(peak_area_2_combo)
                                peaks_dur_2_combo.append(peak_dur_2_combo) 
                                peaks_lp_1_combo.append((start, stop)) 

                                au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)


                            # SINGLE AU1
                            else:

                                # compute features
                                peak_prom_1_simple = float(np.max(current_peak_1))
                                peak_int_1_simple = float(np.mean(current_peak_1))
                                peak_area_1_simple = float(np.sum(current_peak_1))
                                peak_dur_1_simple = float(len(current_peak_1)/fr)

                                # save features
                                peaks_int_1_simple.append(peak_int_1_simple)
                                peaks_prom_1_simple.append(peak_prom_1_simple)
                                peaks_area_1_simple.append(peak_area_1_simple)
                                peaks_dur_1_simple.append(peak_dur_1_simple) 
                                peaks_lp_1_simple.append((start, stop))  

                                au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)


        peaks_1_simple_dict = dict = {'int_avg_simple_'+au1_name: peaks_int_1_simple, 'prominence_simple_'+au1_name: peaks_prom_1_simple, 
        'area_simple_'+au1_name: peaks_area_1_simple,'duration_simple_'+au1_name: peaks_dur_1_simple, 'peak_lp_simple_'+au1_name:peaks_lp_1_simple}

        peaks_1_combo_dict = dict = {'int_avg_combo_'+au1_name: peaks_int_1_combo, 'prominence_combo_'+au1_name: peaks_prom_1_combo, 
        'area_combo_'+au1_name: peaks_area_1_combo,'duration_combo_'+au1_name: peaks_dur_1_combo, 'peak_lp_combo_'+au1_name:peaks_lp_1_combo,
        'int_avg_combo_'+au2_name: peaks_int_2_combo, 'prominence_combo_'+au2_name: peaks_prom_2_combo, 
        'area_combo_'+au2_name: peaks_area_2_combo,'duration_combo_'+au2_name: peaks_dur_2_combo}


        
    elif len(list(aus_dict.keys()))==3:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au3_name = str(list(aus_dict.keys())[2])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        au3_df = list(aus_dict.values())[2]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak3_hp = list(peak_hp_dict.values())[2]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]
        peak3_lp = list(peak_lp_dict.values())[2]
 

        # FIRST SIMPLE COMBO AU1

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]


                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]
                                
                                #check for au3 peaks in au1+au2 combo
                                if any( start < elem_3 < stop for elem_3 in peak3_hp):


                                    current_peak_3 = au3_df.iloc[start:stop]
                                    #get only the values of the au3 peak different from 0
                                    current_peak_3 = current_peak_3.loc[(current_peak_3 > height).any(axis=1)]


                                    # save au1 features for combo
                                    peak_prom_1_combo = float(np.max(current_peak_1))
                                    peak_int_1_combo = float(np.mean(current_peak_1))
                                    peak_area_1_combo = float(np.sum(current_peak_1))
                                    peak_dur_1_combo = float(len(current_peak_1)/fr)

                                    # save au2 features for combo
                                    peak_prom_2_combo = float(np.max(current_peak_2))
                                    peak_int_2_combo = float(np.mean(current_peak_2))
                                    peak_area_2_combo = float(np.sum(current_peak_2))
                                    peak_dur_2_combo = float(len(current_peak_2)/fr)


                                    # save au3 features for combo
                                    peak_prom_3_combo = float(np.max(current_peak_3))
                                    peak_int_3_combo = float(np.mean(current_peak_3))
                                    peak_area_3_combo = float(np.sum(current_peak_3))
                                    peak_dur_3_combo = float(len(current_peak_3)/fr)


                                    # add features to list
                                    peaks_int_1_combo.append(peak_int_1_combo)
                                    peaks_prom_1_combo.append(peak_prom_1_combo)
                                    peaks_area_1_combo.append(peak_area_1_combo)
                                    peaks_dur_1_combo.append(peak_dur_1_combo) 
                                    peaks_int_2_combo.append(peak_int_2_combo)
                                    peaks_prom_2_combo.append(peak_prom_2_combo)
                                    peaks_area_2_combo.append(peak_area_2_combo)
                                    peaks_dur_2_combo.append(peak_dur_2_combo) 
                                    peaks_int_3_combo.append(peak_int_3_combo)
                                    peaks_prom_3_combo.append(peak_prom_3_combo)
                                    peaks_area_3_combo.append(peak_area_3_combo)
                                    peaks_dur_3_combo.append(peak_dur_3_combo) 
                                    peaks_lp_1_combo.append((start, stop)) 

                                    au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

                                
                                else:

                                    # save au1 features for simple
                                    peak_prom_1_simple = float(np.max(current_peak_1))
                                    peak_int_1_simple = float(np.mean(current_peak_1))
                                    peak_area_1_simple = float(np.sum(current_peak_1))
                                    peak_dur_1_simple = float(len(current_peak_1)/fr)

                                    # save au2 features for simple
                                    peak_prom_2_simple = float(np.max(current_peak_2))
                                    peak_int_2_simple = float(np.mean(current_peak_2))
                                    peak_area_2_simple = float(np.sum(current_peak_2))
                                    peak_dur_2_simple = float(len(current_peak_2)/fr)

                                    # add features to list
                                    peaks_int_1_simple.append(peak_int_1_simple)
                                    peaks_prom_1_simple.append(peak_prom_1_simple)
                                    peaks_area_1_simple.append(peak_area_1_simple)
                                    peaks_dur_1_simple.append(peak_dur_1_simple) 
                                    peaks_int_2_simple.append(peak_int_2_simple)
                                    peaks_prom_2_simple.append(peak_prom_2_simple)
                                    peaks_area_2_simple.append(peak_area_2_simple)
                                    peaks_dur_2_simple.append(peak_dur_2_simple) 
                                    peaks_lp_1_simple.append((start, stop)) 

                                    au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

        
        peaks_1_simple_dict = dict = {'int_avg_simple_'+au1_name: peaks_int_1_simple, 'prominence_simple_'+au1_name: peaks_prom_1_simple, 
        'area_simple_'+au1_name: peaks_area_1_simple,'duration_simple_'+au1_name: peaks_dur_1_simple, 'peak_lp_simple_'+au1_name:peaks_lp_1_simple,
        'int_avg_simple_'+au2_name: peaks_int_2_simple, 'prominence_simple_'+au2_name: peaks_prom_2_simple, 
        'area_simple_'+au2_name: peaks_area_2_simple,'duration_simple_'+au2_name: peaks_dur_2_simple}

        peaks_1_combo_dict = dict = {'int_avg_combo_'+au1_name: peaks_int_1_combo, 'prominence_combo_'+au1_name: peaks_prom_1_combo, 
        'area_combo_'+au1_name: peaks_area_1_combo,'duration_combo_'+au1_name: peaks_dur_1_combo, 'peak_lp_combo_'+au1_name:peaks_lp_1_combo,
        'int_avg_combo_'+au2_name: peaks_int_2_combo, 'prominence_combo_'+au2_name: peaks_prom_2_combo, 
        'area_combo_'+au2_name: peaks_area_2_combo,'duration_combo_'+au2_name: peaks_dur_2_combo,
        'int_avg_combo_'+au3_name: peaks_int_3_combo, 'prominence_combo_'+au3_name: peaks_prom_3_combo, 
        'area_combo_'+au3_name: peaks_area_3_combo,'duration_combo_'+au3_name: peaks_dur_3_combo}
        

    
    elif len(list(aus_dict.keys()))==4:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au3_name = str(list(aus_dict.keys())[2])
        au4_name = str(list(aus_dict.keys())[3])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        au3_df = list(aus_dict.values())[2]
        au4_df = list(aus_dict.values())[3]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak3_hp = list(peak_hp_dict.values())[2]
        peak4_hp = list(peak_hp_dict.values())[3]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]
        peak3_lp = list(peak_lp_dict.values())[2]
        peak4_lp = list(peak_lp_dict.values())[3]



        # forst SIMéLE COMBO AU1

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]


                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp) and any( start < elem_3 < stop for elem_3 in peak3_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]

                                current_peak_3 = au3_df.iloc[start:stop]
                                #get only the values of the au3 peak different from 0
                                current_peak_3 = current_peak_3.loc[(current_peak_3 > height).any(axis=1)]
                                
                                #check for au3 peaks in au1+au2 combo
                                if any( start < elem_4 < stop for elem_4 in peak4_hp):


                                    current_peak_4 = au4_df.iloc[start:stop]
                                    #get only the values of the au4 peak different from 0
                                    current_peak_4 = current_peak_4.loc[(current_peak_4 > height).any(axis=1)]


                                    # save au1 features for combo
                                    peak_prom_1_combo = float(np.max(current_peak_1))
                                    peak_int_1_combo = float(np.mean(current_peak_1))
                                    peak_area_1_combo = float(np.sum(current_peak_1))
                                    peak_dur_1_combo = float(len(current_peak_1)/fr)

                                    # save au2 features for combo
                                    peak_prom_2_combo = float(np.max(current_peak_2))
                                    peak_int_2_combo = float(np.mean(current_peak_2))
                                    peak_area_2_combo = float(np.sum(current_peak_2))
                                    peak_dur_2_combo = float(len(current_peak_2)/fr)


                                    # save au3 features for combo
                                    peak_prom_3_combo = float(np.max(current_peak_3))
                                    peak_int_3_combo = float(np.mean(current_peak_3))
                                    peak_area_3_combo = float(np.sum(current_peak_3))
                                    peak_dur_3_combo = float(len(current_peak_3)/fr)

                                    # save au4 features for combo
                                    peak_prom_4_combo = float(np.max(current_peak_4))
                                    peak_int_4_combo = float(np.mean(current_peak_4))
                                    peak_area_4_combo = float(np.sum(current_peak_4))
                                    peak_dur_4_combo = float(len(current_peak_4)/fr)


                                    # add features to list
                                    peaks_int_1_combo.append(peak_int_1_combo)
                                    peaks_prom_1_combo.append(peak_prom_1_combo)
                                    peaks_area_1_combo.append(peak_area_1_combo)
                                    peaks_dur_1_combo.append(peak_dur_1_combo) 
                                    peaks_int_2_combo.append(peak_int_2_combo)
                                    peaks_prom_2_combo.append(peak_prom_2_combo)
                                    peaks_area_2_combo.append(peak_area_2_combo)
                                    peaks_dur_2_combo.append(peak_dur_2_combo) 
                                    peaks_int_3_combo.append(peak_int_3_combo)
                                    peaks_prom_3_combo.append(peak_prom_3_combo)
                                    peaks_area_3_combo.append(peak_area_3_combo)
                                    peaks_dur_3_combo.append(peak_dur_3_combo) 
                                    peaks_int_4_combo.append(peak_int_4_combo)
                                    peaks_prom_4_combo.append(peak_prom_4_combo)
                                    peaks_area_4_combo.append(peak_area_4_combo)
                                    peaks_dur_4_combo.append(peak_dur_4_combo) 
                                    peaks_lp_1_combo.append((start, stop)) 

                                    au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

                                
                                else:

                                    # save au1 features for simple
                                    peak_prom_1_simple = float(np.max(current_peak_1))
                                    peak_int_1_simple = float(np.mean(current_peak_1))
                                    peak_area_1_simple = float(np.sum(current_peak_1))
                                    peak_dur_1_simple = float(len(current_peak_1)/fr)

                                    # save au2 features for simple
                                    peak_prom_2_simple = float(np.max(current_peak_2))
                                    peak_int_2_simple = float(np.mean(current_peak_2))
                                    peak_area_2_simple = float(np.sum(current_peak_2))
                                    peak_dur_2_simple = float(len(current_peak_2)/fr)

                                    # save au2 features for simple
                                    peak_prom_3_simple = float(np.max(current_peak_3))
                                    peak_int_3_simple = float(np.mean(current_peak_3))
                                    peak_area_3_simple = float(np.sum(current_peak_3))
                                    peak_dur_3_simple = float(len(current_peak_3)/fr)

                                    # add features to list
                                    peaks_int_1_simple.append(peak_int_1_simple)
                                    peaks_prom_1_simple.append(peak_prom_1_simple)
                                    peaks_area_1_simple.append(peak_area_1_simple)
                                    peaks_dur_1_simple.append(peak_dur_1_simple) 
                                    peaks_int_2_simple.append(peak_int_2_simple)
                                    peaks_prom_2_simple.append(peak_prom_2_simple)
                                    peaks_area_2_simple.append(peak_area_2_simple)
                                    peaks_dur_2_simple.append(peak_dur_2_simple) 
                                    peaks_int_3_simple.append(peak_int_3_simple)
                                    peaks_prom_3_simple.append(peak_prom_3_simple)
                                    peaks_area_3_simple.append(peak_area_3_simple)
                                    peaks_dur_3_simple.append(peak_dur_3_simple) 
                                    peaks_lp_1_simple.append((start, stop)) 

                                    au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

            

        peaks_1_simple_dict = dict = {'int_avg_simple_'+au1_name: peaks_int_1_simple, 'prominence_simple_'+au1_name: peaks_prom_1_simple, 
        'area_simple_'+au1_name: peaks_area_1_simple,'duration_simple_'+au1_name: peaks_dur_1_simple, 'peak_lp_simple_'+au1_name:peaks_lp_1_simple,
        'int_avg_simple_'+au2_name: peaks_int_2_simple, 'prominence_simple_'+au2_name: peaks_prom_2_simple, 
        'area_simple_'+au2_name: peaks_area_2_simple,'duration_simple_'+au2_name: peaks_dur_2_simple,
        'int_avg_simple_'+au3_name: peaks_int_3_simple, 'prominence_simple_'+au3_name: peaks_prom_3_simple, 
        'area_simple_'+au3_name: peaks_area_3_simple,'duration_simple_'+au3_name: peaks_dur_3_simple}

        peaks_1_combo_dict = dict = {'int_avg_combo_'+au1_name: peaks_int_1_combo, 'prominence_combo_'+au1_name: peaks_prom_1_combo, 
        'area_combo_'+au1_name: peaks_area_1_combo,'duration_combo_'+au1_name: peaks_dur_1_combo, 'peak_lp_combo_'+au1_name:peaks_lp_1_combo,
        'int_avg_combo_'+au2_name: peaks_int_2_combo, 'prominence_combo_'+au2_name: peaks_prom_2_combo, 
        'area_combo_'+au2_name: peaks_area_2_combo,'duration_combo_'+au2_name: peaks_dur_2_combo,
        'int_avg_combo_'+au3_name: peaks_int_3_combo, 'prominence_combo_'+au3_name: peaks_prom_3_combo, 
        'area_combo_'+au3_name: peaks_area_3_combo,'duration_combo_'+au3_name: peaks_dur_3_combo,
            'int_avg_combo_'+au4_name: peaks_int_4_combo, 'prominence_combo_'+au4_name: peaks_prom_4_combo, 
        'area_combo_'+au4_name: peaks_area_4_combo,'duration_combo_'+au4_name: peaks_dur_4_combo}



    if len(aus_dict.keys())>4:

        print('More than 4 AUs!')


    peaks_1_simple_df = pd.DataFrame(peaks_1_simple_dict)
    peaks_1_combo_df = pd.DataFrame(peaks_1_combo_dict)



    return peaks_1_simple_df, au_bin_1_simple, peaks_1_combo_df, au_bin_1_combo   







## get features from smoothed signal and peaks for COMBO AU with AVERAGE of AUS
def ComputeAverageComboAUPeaksFeatures( df, aus_dict, peak_hp_dict, peak_lp_dict, height, fr):

    # get frame rate

    times = np.arange(len(df))/fr
    

    #SIMPLE
    peaks_int_simple = []
    peaks_prom_simple = []
    peaks_area_simple = []
    peaks_dur_simple = []


    peaks_int_1_simple = []
    peaks_prom_1_simple = []
    peaks_area_1_simple = []
    peaks_dur_1_simple = []
    peaks_lp_1_simple = []

    peaks_int_2_simple = []
    peaks_prom_2_simple = []
    peaks_area_2_simple = []
    peaks_dur_2_simple = []
    peaks_lp_2_simple = []

    peaks_int_3_simple = []
    peaks_prom_3_simple = []
    peaks_area_3_simple = []
    peaks_dur_3_simple = []
    peaks_lp_3_simple = []

    peaks_int_4_simple = []
    peaks_prom_4_simple = []
    peaks_area_4_simple = []
    peaks_dur_4_simple = []
    peaks_lp_4_simple = []

    #COMBO
    peaks_int_combo = []
    peaks_prom_combo = []
    peaks_area_combo = []
    peaks_dur_combo = []


    peaks_int_1_combo = []
    peaks_prom_1_combo = []
    peaks_area_1_combo = []
    peaks_dur_1_combo = []
    peaks_lp_1_combo = []

    peaks_int_2_combo = []
    peaks_prom_2_combo = []
    peaks_area_2_combo = []
    peaks_dur_2_combo = []
    peaks_lp_2_combo = []

    peaks_int_3_combo = []
    peaks_prom_3_combo = []
    peaks_area_3_combo = []
    peaks_dur_3_combo = []
    peaks_lp_3_combo = []

    peaks_int_4_combo = []
    peaks_prom_4_combo = []
    peaks_area_4_combo = []
    peaks_dur_4_combo = []
    peaks_lp_4_combo = []

    au_bin_1_simple = pd.DataFrame(np.array([0] * len(df)))    
    au_bin_1_combo = pd.DataFrame(np.array([0] * len(df)))


    print('Number of given AUs: ',len(list(aus_dict.keys())))

    if len(list(aus_dict.keys()))==2:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]


        # SIMPLE

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]

                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]

                                # save au1 features for combo
                                peak_prom_1_combo = float(np.max(current_peak_1))
                                peak_int_1_combo = float(np.mean(current_peak_1))
                                peak_area_1_combo = float(np.sum(current_peak_1))
                                peak_dur_1_combo = float(len(current_peak_1)/fr)

                                # save au2 features for combo
                                peak_prom_2_combo = float(np.max(current_peak_2))
                                peak_int_2_combo = float(np.mean(current_peak_2))
                                peak_area_2_combo = float(np.sum(current_peak_2))
                                peak_dur_2_combo = float(len(current_peak_2)/fr)

                                # save average
                                peak_prom_combo = np.mean(peak_prom_1_combo+peak_prom_2_combo)/2
                                peak_int_combo = np.mean(peak_int_1_combo+peak_int_2_combo)/2
                                peak_area_combo = np.mean(peak_area_1_combo+peak_area_2_combo)/2
                                peak_dur_combo = np.mean(peak_dur_1_combo+peak_dur_2_combo)/2

                                # add features to list
                                peaks_int_1_combo.append(peak_int_1_combo)
                                peaks_prom_1_combo.append(peak_prom_1_combo)
                                peaks_area_1_combo.append(peak_area_1_combo)
                                peaks_dur_1_combo.append(peak_dur_1_combo) 
                                peaks_int_2_combo.append(peak_int_2_combo)
                                peaks_prom_2_combo.append(peak_prom_2_combo)
                                peaks_area_2_combo.append(peak_area_2_combo)
                                peaks_dur_2_combo.append(peak_dur_2_combo) 
                                peaks_int_combo.append(peak_int_combo)
                                peaks_prom_combo.append(peak_prom_combo)
                                peaks_area_combo.append(peak_area_combo)
                                peaks_dur_combo.append(peak_dur_combo) 
                                peaks_lp_1_combo.append((start, stop)) 

                                au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)


                            # SINGLE AU1
                            else:

                                # compute features
                                peak_prom_1_simple = float(np.max(current_peak_1))
                                peak_int_1_simple = float(np.mean(current_peak_1))
                                peak_area_1_simple = float(np.sum(current_peak_1))
                                peak_dur_1_simple = float(len(current_peak_1)/fr)

                                # save features
                                peaks_int_1_simple.append(peak_int_1_simple)
                                peaks_prom_1_simple.append(peak_prom_1_simple)
                                peaks_area_1_simple.append(peak_area_1_simple)
                                peaks_dur_1_simple.append(peak_dur_1_simple) 
                                peaks_lp_1_simple.append((start, stop))  

                                au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)


        peaks_1_simple_dict = dict = {'int_avg_simple': peaks_int_1_simple, 'prominence_simple': peaks_prom_1_simple, 
        'area_simple': peaks_area_1_simple,'duration_simple': peaks_dur_1_simple, 'peak_lp_simple':peaks_lp_1_simple}

        peaks_1_combo_dict = dict = {'peak_lp_combo':peaks_lp_1_combo,
        'int_avg_combo': peaks_int_combo, 
        'prominence_combo': peaks_prom_combo, 'area_combo': peaks_area_combo,'duration_combo': peaks_dur_combo}


        
    elif len(list(aus_dict.keys()))==3:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au3_name = str(list(aus_dict.keys())[2])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        au3_df = list(aus_dict.values())[2]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak3_hp = list(peak_hp_dict.values())[2]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]
        peak3_lp = list(peak_lp_dict.values())[2]
 

        # FIRST SIMPLE COMBO AU1

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]


                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]
                                
                                #check for au3 peaks in au1+au2 combo
                                if any( start < elem_3 < stop for elem_3 in peak3_hp):


                                    current_peak_3 = au3_df.iloc[start:stop]
                                    #get only the values of the au3 peak different from 0
                                    current_peak_3 = current_peak_3.loc[(current_peak_3 > height).any(axis=1)]


                                    # save au1 features for combo
                                    peak_prom_1_combo = float(np.max(current_peak_1))
                                    peak_int_1_combo = float(np.mean(current_peak_1))
                                    peak_area_1_combo = float(np.sum(current_peak_1))
                                    peak_dur_1_combo = float(len(current_peak_1)/fr)

                                    # save au2 features for combo
                                    peak_prom_2_combo = float(np.max(current_peak_2))
                                    peak_int_2_combo = float(np.mean(current_peak_2))
                                    peak_area_2_combo = float(np.sum(current_peak_2))
                                    peak_dur_2_combo = float(len(current_peak_2)/fr)


                                    # save au3 features for combo
                                    peak_prom_3_combo = float(np.max(current_peak_3))
                                    peak_int_3_combo = float(np.mean(current_peak_3))
                                    peak_area_3_combo = float(np.sum(current_peak_3))
                                    peak_dur_3_combo = float(len(current_peak_3)/fr)


                                    # save average
                                    peak_prom_combo = np.mean(peak_prom_1_combo+peak_prom_2_combo+peak_prom_3_combo)/3
                                    peak_int_combo = np.mean(peak_int_1_combo+peak_int_2_combo+peak_int_3_combo)/3
                                    peak_area_combo = np.mean(peak_area_1_combo+peak_area_2_combo+peak_area_3_combo)/3
                                    peak_dur_combo = np.mean(peak_dur_1_combo+peak_dur_2_combo+peak_dur_3_combo)/3


                                    # add features to list
                                    peaks_int_1_combo.append(peak_int_1_combo)
                                    peaks_prom_1_combo.append(peak_prom_1_combo)
                                    peaks_area_1_combo.append(peak_area_1_combo)
                                    peaks_dur_1_combo.append(peak_dur_1_combo) 
                                    peaks_int_2_combo.append(peak_int_2_combo)
                                    peaks_prom_2_combo.append(peak_prom_2_combo)
                                    peaks_area_2_combo.append(peak_area_2_combo)
                                    peaks_dur_2_combo.append(peak_dur_2_combo) 
                                    peaks_int_3_combo.append(peak_int_3_combo)
                                    peaks_prom_3_combo.append(peak_prom_3_combo)
                                    peaks_area_3_combo.append(peak_area_3_combo)
                                    peaks_dur_3_combo.append(peak_dur_3_combo) 
                                    peaks_int_combo.append(peak_int_combo)
                                    peaks_prom_combo.append(peak_prom_combo)
                                    peaks_area_combo.append(peak_area_combo)
                                    peaks_dur_combo.append(peak_dur_combo)
                                    peaks_lp_1_combo.append((start, stop)) 

                                    au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

                                
                                else:

                                    # save au1 features for simple
                                    peak_prom_1_simple = float(np.max(current_peak_1))
                                    peak_int_1_simple = float(np.mean(current_peak_1))
                                    peak_area_1_simple = float(np.sum(current_peak_1))
                                    peak_dur_1_simple = float(len(current_peak_1)/fr)

                                    # save au2 features for simple
                                    peak_prom_2_simple = float(np.max(current_peak_2))
                                    peak_int_2_simple = float(np.mean(current_peak_2))
                                    peak_area_2_simple = float(np.sum(current_peak_2))
                                    peak_dur_2_simple = float(len(current_peak_2)/fr)

                                        # save average
                                    peak_prom_simple = np.mean(peak_prom_1_simple+peak_prom_2_simple)/2
                                    peak_int_simple = np.mean(peak_int_1_simple+peak_int_2_simple)/2
                                    peak_area_simple = np.mean(peak_area_1_simple+peak_area_2_simple)/2
                                    peak_dur_simple = np.mean(peak_dur_1_simple+peak_dur_2_simple)/2

                                    # add features to list
                                    peaks_int_1_simple.append(peak_int_1_simple)
                                    peaks_prom_1_simple.append(peak_prom_1_simple)
                                    peaks_area_1_simple.append(peak_area_1_simple)
                                    peaks_dur_1_simple.append(peak_dur_1_simple) 
                                    peaks_int_2_simple.append(peak_int_2_simple)
                                    peaks_prom_2_simple.append(peak_prom_2_simple)
                                    peaks_area_2_simple.append(peak_area_2_simple)
                                    peaks_dur_2_simple.append(peak_dur_2_simple) 
                                    peaks_int_simple.append(peak_int_simple)
                                    peaks_prom_simple.append(peak_prom_simple)
                                    peaks_area_simple.append(peak_area_simple)
                                    peaks_dur_simple.append(peak_dur_simple) 
                                    peaks_lp_1_simple.append((start, stop)) 

                                    au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

        
        peaks_1_simple_dict = dict = {'peak_lp_simple':peaks_lp_1_simple,
        'int_avg_simple': peaks_int_simple, 
        'prominence_simple': peaks_prom_simple, 'area_simple': peaks_area_simple,'duration_simple': peaks_dur_simple}

        peaks_1_combo_dict = dict = {'peak_lp_combo':peaks_lp_1_combo,
        'int_avg_combo': peaks_int_combo, 
        'prominence_combo': peaks_prom_combo, 'area_combo': peaks_area_combo,'duration_combo': peaks_dur_combo}
        

    
    elif len(list(aus_dict.keys()))==4:

        au1_name = str(list(aus_dict.keys())[0])
        au2_name = str(list(aus_dict.keys())[1])
        au3_name = str(list(aus_dict.keys())[2])
        au4_name = str(list(aus_dict.keys())[3])
        au1_df = list(aus_dict.values())[0]
        au2_df = list(aus_dict.values())[1]
        au3_df = list(aus_dict.values())[2]
        au4_df = list(aus_dict.values())[3]
        peak1_hp = list(peak_hp_dict.values())[0]
        peak2_hp = list(peak_hp_dict.values())[1]
        peak3_hp = list(peak_hp_dict.values())[2]
        peak4_hp = list(peak_hp_dict.values())[3]
        peak1_lp = list(peak_lp_dict.values())[0]
        peak2_lp = list(peak_lp_dict.values())[1]
        peak3_lp = list(peak_lp_dict.values())[2]
        peak4_lp = list(peak_lp_dict.values())[3]



        # forst SIMéLE COMBO AU1

        if len(peak1_hp) != 0:
            for start,stop in zip(peak1_lp, peak1_lp[1:]):
                
                #iterate over high peaks
                for elem_1 in peak1_hp:
                    #selecting a current peak of au1
                    if start < elem_1 < stop:  
                        #filter by minimun duration of 0.5 sec
                        if float((stop - start)/fr) > 0.5: 
                            #selecting current peak of au1
                            current_peak_1 = au1_df.iloc[start:stop]
                            current_peak_1 = current_peak_1.loc[(current_peak_1 > height).any(axis=1)]


                            #check if there are au2 peaks in the current au1 peaks (COMBO)
                            if any( start < elem_2 < stop for elem_2 in peak2_hp) and any( start < elem_3 < stop for elem_3 in peak3_hp):

                                current_peak_2 = au2_df.iloc[start:stop]
                                #get only the values of the au2 peak different from 0
                                current_peak_2 = current_peak_2.loc[(current_peak_2 > height).any(axis=1)]

                                current_peak_3 = au3_df.iloc[start:stop]
                                #get only the values of the au3 peak different from 0
                                current_peak_3 = current_peak_3.loc[(current_peak_3 > height).any(axis=1)]
                                
                                #check for au3 peaks in au1+au2 combo
                                if any( start < elem_4 < stop for elem_4 in peak4_hp):


                                    current_peak_4 = au4_df.iloc[start:stop]
                                    #get only the values of the au4 peak different from 0
                                    current_peak_4 = current_peak_4.loc[(current_peak_4 > height).any(axis=1)]


                                    # save au1 features for combo
                                    peak_prom_1_combo = float(np.max(current_peak_1))
                                    peak_int_1_combo = float(np.mean(current_peak_1))
                                    peak_area_1_combo = float(np.sum(current_peak_1))
                                    peak_dur_1_combo = float(len(current_peak_1)/fr)

                                    # save au2 features for combo
                                    peak_prom_2_combo = float(np.max(current_peak_2))
                                    peak_int_2_combo = float(np.mean(current_peak_2))
                                    peak_area_2_combo = float(np.sum(current_peak_2))
                                    peak_dur_2_combo = float(len(current_peak_2)/fr)


                                    # save au3 features for combo
                                    peak_prom_3_combo = float(np.max(current_peak_3))
                                    peak_int_3_combo = float(np.mean(current_peak_3))
                                    peak_area_3_combo = float(np.sum(current_peak_3))
                                    peak_dur_3_combo = float(len(current_peak_3)/fr)

                                    # save au4 features for combo
                                    peak_prom_4_combo = float(np.max(current_peak_4))
                                    peak_int_4_combo = float(np.mean(current_peak_4))
                                    peak_area_4_combo = float(np.sum(current_peak_4))
                                    peak_dur_4_combo = float(len(current_peak_4)/fr)

                                    # save average
                                    peak_prom_combo = np.mean(peak_prom_1_combo+peak_prom_2_combo+peak_prom_3_combo+peak_prom_4_combo)/4
                                    peak_int_combo = np.mean(peak_int_1_combo+peak_int_2_combo+peak_int_3_combo+peak_int_4_combo)/4
                                    peak_area_combo = np.mean(peak_area_1_combo+peak_area_2_combo+peak_area_3_combo+peak_area_4_combo)/4
                                    peak_dur_combo = np.mean(peak_dur_1_combo+peak_dur_2_combo+peak_dur_3_combo+peak_dur_4_combo)/4


                                    # add features to list
                                    peaks_int_1_combo.append(peak_int_1_combo)
                                    peaks_prom_1_combo.append(peak_prom_1_combo)
                                    peaks_area_1_combo.append(peak_area_1_combo)
                                    peaks_dur_1_combo.append(peak_dur_1_combo) 
                                    peaks_int_2_combo.append(peak_int_2_combo)
                                    peaks_prom_2_combo.append(peak_prom_2_combo)
                                    peaks_area_2_combo.append(peak_area_2_combo)
                                    peaks_dur_2_combo.append(peak_dur_2_combo) 
                                    peaks_int_3_combo.append(peak_int_3_combo)
                                    peaks_prom_3_combo.append(peak_prom_3_combo)
                                    peaks_area_3_combo.append(peak_area_3_combo)
                                    peaks_dur_3_combo.append(peak_dur_3_combo) 
                                    peaks_int_4_combo.append(peak_int_4_combo)
                                    peaks_prom_4_combo.append(peak_prom_4_combo)
                                    peaks_area_4_combo.append(peak_area_4_combo)
                                    peaks_dur_4_combo.append(peak_dur_4_combo) 
                                    peaks_int_combo.append(peak_int_combo)
                                    peaks_prom_combo.append(peak_prom_combo)
                                    peaks_area_combo.append(peak_area_combo)
                                    peaks_dur_combo.append(peak_dur_combo) 
                                    peaks_lp_1_combo.append((start, stop)) 

                                    au_bin_1_combo.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

                                
                                else:

                                    # save au1 features for simple
                                    peak_prom_1_simple = float(np.max(current_peak_1))
                                    peak_int_1_simple = float(np.mean(current_peak_1))
                                    peak_area_1_simple = float(np.sum(current_peak_1))
                                    peak_dur_1_simple = float(len(current_peak_1)/fr)

                                    # save au2 features for simple
                                    peak_prom_2_simple = float(np.max(current_peak_2))
                                    peak_int_2_simple = float(np.mean(current_peak_2))
                                    peak_area_2_simple = float(np.sum(current_peak_2))
                                    peak_dur_2_simple = float(len(current_peak_2)/fr)

                                    # save au2 features for simple
                                    peak_prom_3_simple = float(np.max(current_peak_3))
                                    peak_int_3_simple = float(np.mean(current_peak_3))
                                    peak_area_3_simple = float(np.sum(current_peak_3))
                                    peak_dur_3_simple = float(len(current_peak_3)/fr)

                                    # save average
                                    peak_prom_simple = np.mean(peak_prom_1_simple+peak_prom_2_simple+peak_prom_3_simple)
                                    peak_int_simple = np.mean(peak_int_1_simple+peak_int_2_simple+peak_int_3_simple)
                                    peak_area_simple = np.mean(peak_area_1_simple+peak_area_2_simple+peak_area_3_simple)
                                    peak_dur_simple = np.mean(peak_dur_1_simple+peak_dur_2_simple+peak_dur_3_simple)

                                    # add features to list
                                    peaks_int_1_simple.append(peak_int_1_simple)
                                    peaks_prom_1_simple.append(peak_prom_1_simple)
                                    peaks_area_1_simple.append(peak_area_1_simple)
                                    peaks_dur_1_simple.append(peak_dur_1_simple) 
                                    peaks_int_2_simple.append(peak_int_2_simple)
                                    peaks_prom_2_simple.append(peak_prom_2_simple)
                                    peaks_area_2_simple.append(peak_area_2_simple)
                                    peaks_dur_2_simple.append(peak_dur_2_simple) 
                                    peaks_int_3_simple.append(peak_int_3_simple)
                                    peaks_prom_3_simple.append(peak_prom_3_simple)
                                    peaks_area_3_simple.append(peak_area_3_simple)
                                    peaks_dur_3_simple.append(peak_dur_3_simple) 
                                    peaks_int_simple.append(peak_int_simple)
                                    peaks_prom_simple.append(peak_prom_simple)
                                    peaks_area_simple.append(peak_area_simple)
                                    peaks_dur_simple.append(peak_dur_simple) 
                                    peaks_lp_1_simple.append((start, stop)) 

                                    au_bin_1_simple.iloc[start:stop] = np.where(au1_df.iloc[start:stop] > 0, 1, 0)

            

        peaks_1_simple_dict = dict = {'peak_lp_simple':peaks_lp_1_simple,'int_avg_simple': peaks_int_simple, 
        'prominence_simple': peaks_prom_simple, 'area_simple': peaks_area_simple,'duration_simple': peaks_dur_simple}

        peaks_1_combo_dict = dict = {'peak_lp_combo':peaks_lp_1_combo,
        'int_avg_combo': peaks_int_combo, 
        'prominence_combo': peaks_prom_combo, 'area_combo': peaks_area_combo,'duration_combo': peaks_dur_combo}



    if len(aus_dict.keys())>4:

        print('More than 4 AUs!')


    peaks_1_simple_df = pd.DataFrame(peaks_1_simple_dict)
    peaks_1_combo_df = pd.DataFrame(peaks_1_combo_dict)



    return peaks_1_simple_df, au_bin_1_simple, peaks_1_combo_df, au_bin_1_combo   
