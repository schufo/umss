'''
This script generates ground truth f0 files ready for robustness testing from crepe's raw output files.
'''

from locale import currency
import pandas as pd
import glob
import math
import numpy as np

def find_nth(haystack,needle, n):
            start = haystack.find(needle)
            while start >= 0 and n > 1:
                start = haystack.find(needle,start+len(needle))
                n -=1
            return start

def process_confidence(df=pd.DataFrame(),conf_threshold=0.6):
    arr=df.to_numpy()
    for idx in range(arr.shape[0]):
        if arr[idx][2]<=conf_threshold: #2 is confidence
            arr[idx][1]=None            #1 is frequency
    df=pd.DataFrame(arr, columns=['time', 'frequency', 'confidence'])
    return(df)

def time_to_int(df=pd.DataFrame()):
    # arr=df.to_numpy()
    # for idx in range(arr.shape[0]):
    #     # time=arr[idx][0]
    #     arr[idx][0]=int(((arr[idx][0]*1000+1)//16)*16)
    # df=pd.DataFrame(arr, columns=['time', 'frequency', 'confidence'])
    df['time']=df['time'].apply(lambda time : int((time*1000+1)//16)*16) #there were a few weird lines that WEREN'T step 16 and instead 15 then 17.
    # df=df.astype({'time':int}) # J'arrives pas à convertir en int :(
    return(df)

def convert_raw_crepe_to_gtf0(conf_threshold=0.6):
    '''Generates ground truth f0 files from crepe's raw output files with a certain confidence threshold.'''
    song_names={
        "rossinyol" : "El_Rossinyol",
        "locus" : "Locus_Iste",
        "nino" : "Nino_Dios"
    }  
    for c in ["crepe_centered","crepe_no-centering"]:
        for song in song_names.keys():
            satb_dict={}
            csvs=glob.glob("./{}/{}_*.csv".format(c,song))
            for csv in csvs:
                temp_df=pd.read_csv(csv)
                temp_df=time_to_int(temp_df)
                temp_df=process_confidence(temp_df,conf_threshold)
                if ('time' in satb_dict)== False: #init satb_dict
                    satb_dict['time']=temp_df['time']
                # print("csv :", csv)
                current_voice=csv[find_nth(csv,"_",2)+1].lower()
                if current_voice == 'c' :
                    current_voice= 'a'
                # print("current_voice : ", current_voice)
                satb_dict['{}'.format(current_voice)]=temp_df['frequency']
            satb_df=pd.DataFrame(satb_dict)
            satb_df.index=satb_df.pop('time')
            #reorder columns
            cols=list('satb')
            satb_df=satb_df[cols]
            satb_df.to_csv('./{}/{}_SATB_gtf0.csv'.format(c,song_names[song]))

    print("Converting raw_crepe_to_gtf0 with conf_threshold :", conf_threshold, "done")

'''Main'''
convert_raw_crepe_to_gtf0(conf_threshold=0.702)