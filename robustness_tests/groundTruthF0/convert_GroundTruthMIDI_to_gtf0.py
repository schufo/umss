'''
This script generates ground truth f0 files ready for robustness testing 
from the CSD database's MIDI files.

Loading midi tracks tutorial followed : https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
'''

import pandas as pd
import math
import bisect
import mido

def create_msg_in_ms_df(mid,prnt=False):
    '''Returns a pd.Dataframe with [msg types, midi notes, time in ms] from a mido.MidiFile.'''
    msg_in_ms_list=[]
    time=0
    for msg in mid:
        if not msg.is_meta: # Only pool note messages. Tempo is default 500,000 and same for ticks per beat 48.
            d=msg.dict() #Returns a dictionary containing the attributes of the message. 'type' 'time 'note' are the important ones
            time+=d['time']
            time2=mido.tick2second(time,480,500000) #get time in seconds
            time2=time2*1000000  #Convert to ms
            time2=math.floor(time2) #convert to int
            msg_in_ms_list.append([d['type'],d['note'],time2])

    msg_in_ms_df = pd.DataFrame(msg_in_ms_list, columns=['type','note','time'])
    if prnt:
        print(msg_in_ms_df)
    return (msg_in_ms_df)

def is_note_active(time,msg_in_ms_df):
    '''
    Returns index of active note if there is a note active, otherwise None
    This function assumes msg_in_ms_df alternates msg types note_on and note_off
    '''
    index=bisect.bisect_left(msg_in_ms_df['time'],time) #-1 to properly take the one on the left
    if msg_in_ms_df['type'][index] == 'note_off':
        return(index)
    else:
        return(None) 

def noteToFreq(note):
    ''' Easy MIDI note to frequency convertor https://gist.github.com/YuxiUx/ef84328d95b10d0fcbf537de77b936cd '''
    a = 440 #frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

def create_full_f0_df(msg_in_ms_df,songLength=134,interval=16,prnt=False):
    '''
    Returns a pd.DataFrame with the corresponding notes pooled every 16ms.
    Default interval is 16 ms which corresponds with the CSD item from data.py.
    (there 250 f0 entries in a 4 second CSD data object)
    '''
    time=0
    full_f0_list=[]
    while time<songLength*1000 : #This could be done in a single list comprehension! a massive for thing. But that would be annoying to write and read
        index = is_note_active(time,msg_in_ms_df)
        if index!=None:
            note=msg_in_ms_df['note'][index]
            freq=noteToFreq(note)
            full_f0_list.append([time,freq])
        else:
            full_f0_list.append([time,None])
        time+=interval
    full_f0_df = pd.DataFrame(full_f0_list, columns=['time','freq'])
    if prnt:
        print(full_f0_df)
    return(full_f0_df)

''' Main '''

tagsSongs = [
        ['El_Rossinyol','rossinyol',134],
        ['Locus_Iste','locus',190],
        ['Nino_Dios','nino',103]
        ]

tagsVoices = ['soprano','alto','tenor','bajo']

for tagS in tagsSongs:
    satb_dict={}
    for tagV in tagsVoices:
        mid = mido.MidiFile('../../Datasets/ChoralSingingDataset/{}/midi/{}_{}_midi.mid'.format(tagS[0],tagS[1],tagV), clip=True)
        # print(mid.tracks[0])
        msg_in_ms_df=create_msg_in_ms_df(mid,prnt=False)
        full_f0_df=create_full_f0_df(msg_in_ms_df,songLength=tagS[2],prnt=False)
        # full_f0_df.to_csv('./midi/{}_{}_gtf0.csv'.format(tagS[0],tagV))
        # Concatenating all s a t b dataframes into a single satb dataframe.
        if ('time' in satb_dict)== False: 
            satb_dict['time']=full_f0_df['time']
        satb_dict['{}'.format(tagV)]=full_f0_df['freq']
        print(satb_dict)
    satb_full_f0_df=pd.DataFrame(satb_dict)
    satb_full_f0_df.index=satb_full_f0_df.pop('time')
    satb_full_f0_df.columns=list('satb')
    print(satb_full_f0_df)
    satb_full_f0_df.to_csv('./midi/{}_SATB_gtf0.csv'.format(tagS[0]))