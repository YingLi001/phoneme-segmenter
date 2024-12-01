import pandas as pd

# import utilities
from scipy.io import wavfile
from transformers.utils.dummy_pt_objects import torch_distributed_zero_first
import os
import w2v2_predict
from utils.vad import *

from UnsupSeg import predict as seg_pred

from datasets import load_dataset, load_metric

import newphonetest as npt

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

import sys
from tqdm import tqdm
import shutil

import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics

import soundfile as sf
from multiprocessing.dummy import Pool as ThreadPool
from datetime import date, datetime
import pytz
from praatio import textgrid
import librosa
import csv



bias = 0.5
use_vad = False
#soft clean
use_clean = True
# #False for midpoints, True for onset boundaries
# AIE_evaluation = False
#hard clean
clean_aggressive = True

#   Of an array of 0's and 1's, return the time in seconds when the transitions occur
#   arr: an array of zeroes and ones
#   st: sample length in time (s)
def detectEdges(arr, st):
    tmp = None
    tmp2 = None
    return_array = list()

    for ii in range(len(arr)):
        jj = int(ii)
        if tmp == None:
        #Condtion: we just instantiated the array
            tmp = arr[jj][0]

        #Condition: rising edge
        elif arr[jj][0] != tmp and tmp == 0:
            #Store timing of rising edge in seconds
            tmp2 = jj * st/len(arr)

            tmp = 1

        # Condition: falling edge
        elif arr[jj][0] != tmp and tmp == 1:
            #add the pair of boundaries to the return list
            return_array.append((tmp2, jj * st/len(arr)))
            tmp = 0

    return return_array

#   removes segmentations which are deemed to have
#   occured in spaces of audio where there is no speech
#
#   Signal:     signal data to process
#   sr:         sampling rate
#   tolerance: tolerance in difference in VAD boundaries and Seg boundaries difference
def filterSegmentations(segmentations, signal, sr, tolerance = 0.05):
    vad=VAD(signal, sr, nFFT=2048, win_length=0.025, hop_length=0.01, theshold=0.5)
    vad = vad.astype('int')
    vad = vad.tolist()
    vadEdges = detectEdges(vad, len(signal)/sr)

    filtered_segs = list()
    # Scan through all the segmentations looking for segments which fit withing the boundaries.
    # Works in O(N^2) time because im a pig.
    #filtered_segs.append(0)
    for seg in segmentations:
        for vadBound in vadEdges:
            if vadBound[0]-tolerance <= seg and seg <= vadBound[1]+tolerance:
                filtered_segs.append(seg)
    #filtered_segs.append(len(signal)/sr)

    return filtered_segs

def filterSegmentationsWrapper(wav_path, segmentations):
    signal, sr  = sf.read(wav_path)
    return filterSegmentations(segmentations, signal, sr)

def seg_demo(wav_path, json_path, vad = use_vad, clean = use_clean, clean_agg = clean_aggressive):
    all_alignment_info = {}
    signalData, samplingFrequency  = sf.read(wav_path)

    #Duration of utterance in seconds
    seconds = len(signalData)/samplingFrequency
    print(seconds)
    wp = w2v2_predict.w2v2_predictor()
    wp.set_model(ckpt="path/to/model")

    #tokens = ['h#', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'hh', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', 'l', '[PAD]', '[PAD]', 'ow', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'pau', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'dh', 'dh', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'q', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', 'z', '[PAD]', '[PAD]', 'ix', '[PAD]', '[PAD]', 'tcl', 'tcl', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'tcl', '[PAD]', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'h#']
    tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")
    print('tokens (no collapsed)', tokens)
    collapsed_tokens = wp.pred_wav_with_collapse(wav_path, return_type="phones")
    print('collasped tokens', collapsed_tokens)

    all_alignment_info["tokens"] = tokens
    all_alignment_info["collapsed tokens"]= collapsed_tokens
    #Delta s is half the distance in time between each token
    delta_s = seconds / (2*len(tokens))

    #A list of tokens with time attached. It's called votelist because itll do some voting later on
    timed_token_list = list()

    #instantiate timestamp with one delta s. The distance between each token in time is 2 times delta_s
    timestamp = delta_s

    #This for loop creats a list of tuples with the timing attached to each
    for token in tokens:
        #Timed token is a tuple with the time in the sequence at which it occurs
        timed_token = (token, timestamp)

        #Add timed token to the voter list
        timed_token_list.append(timed_token)

        #Increment the timestamp for the next token
        timestamp = timestamp + 2*delta_s

    # Now keep only the labels worth interpreting
    print("timed_token_list".ljust(40, "="))
    print(timed_token_list)

    filtered_time_token_list = list()
    for tt in timed_token_list:
        if tt[0] != ("[PAD]" or "[UNK]" or "|"):
            filtered_time_token_list.append(tt)
    print(" filtered_time_token_list ".ljust(40, "="))        
    print(filtered_time_token_list)
    all_alignment_info["timetokens"] = filtered_time_token_list
    # Compute Decision Boundaries
    def decision_boundary_calc(filtered_time_token_list, seconds, bias=bias):
        print("Bias:", bias )
        assert 0 <= bias and bias <= 1
        DCB = list()
        for ii in range(len(filtered_time_token_list)):
            if ii == len(filtered_time_token_list) - 1:  # CASE: Last token
                upper = seconds
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            elif ii == 0:  # CASE: First token
                upper = filtered_time_token_list[ii + 1][1] * (bias) + filtered_time_token_list[ii][1] * (1 - bias)
                lower = 0
            else:
                upper = (filtered_time_token_list[ii + 1][1]) * (bias) + (filtered_time_token_list[ii][1]) * (1 - bias)
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            # append phone label, start time, end time tuple
            DCB.append((filtered_time_token_list[ii][0], lower, upper))
        return DCB

    DCB = decision_boundary_calc(filtered_time_token_list, seconds)
    all_alignment_info["decision_boundary"] = DCB
    print(" DCB ".ljust(40, "="))
    print(DCB)
    #Assign Maximal labels
    import json
    try:
        with open('str_unic.json') as str_unic_file:
            str_to_unicode_dict = json.loads(str_unic_file.read())
        Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)
    except Exception:
        print("Failed to open str_unic.json")

    #Lets zip each label with its start and end times
    segList = list()
    [segList.append((DCB[ii][0], DCB[ii][1], DCB[ii][2])) for ii in range(len(DCB))]

    def clean_segs(segList_in, wav_path):
        segList = segList_in.copy()
        print("segList in clean_segs:", segList)
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)

        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))
        print("transitions", transitions)

        index = 0
        for jj in range(len(transitions)):
            found = False
            limitreached = False

            while found == False and limitreached == False:
                if index >= len(segList)-1:
                    limitreached = True
                else:
                    seg_from = segList[index]
                    seg_to = segList[index + 1]

                    #CASE: the two elements are the same, ie seglist: aab, transition: ab, focal:aa, turn seglist into ab
                    if seg_from[0] == seg_to[0] and seg_from[0] == transitions[jj][0]:
                        segList[index] = (segList[index][0], segList[index][1], segList[index+1][2])
                        segList.remove(segList[index+1])


                    #CASE: Transition is found
                    elif seg_from[0] == transitions[jj][0] and seg_to[0] == transitions[jj][1]:
                        found = True
                        index = index + 1

                    else:
                        index = index + 1
                        break
        return segList
    # post-processing: cleaning
    def clean_segs_aggressive(segList_in, wav_path):
        print(" Hard Clean Start ".ljust(20, "="))
        segList = segList_in.copy()
        print("segList in clean_segs:", segList)
        
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)
        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))
        print("transitions", transitions)

        ceiling = len(segList)-2
        jj=0
        finished = False
        while finished == False:
            if jj <= ceiling:
                if segList[jj][0]==segList[jj+1][0]:
                    if not (segList[jj][0],segList[jj+1][0]) in transitions:
                        newSeg = (segList[jj][0], segList[jj][1], segList[jj+1][2])
                        print(newSeg)
                        segList[jj] = newSeg
                        segList.remove(segList[jj+1])
                        ceiling = ceiling-1
                    jj = jj - 1
                jj = jj + 1
            else:
                finished = True
        print(" Hard Clean Finished ".ljust(20, "="))
        return segList
    
    if clean == True:
            print(" Checking the cleaning method ".ljust(40, '='))
            if clean_agg == True:
                print(" Use Hard Clean ".ljust(40, '='))
                segList = clean_segs_aggressive(segList, wav_path)
                
            else:
                print(" Use Soft Clean ".ljust(40, '='))
                segList = clean_segs(segList, wav_path)

    temp = list()
    for seg in segList:
        temp.append((seg[1],seg[2],seg[0]))
    segList = temp
    all_alignment_info["phoneme segmentations"] = segList
    print(" GENERATING JSON FILE ... ".center(40, '*'))
    with open(json_path, "w") as fp:   
        json.dump(all_alignment_info, fp)
    print(" JSON GENERATED !!! ".center(40, '*'))
    return segList

def generate_textgrid(segList, wav_path, tg_path, ML_TG_path):
    print(" GENERATING TEXTGRID ... ".center(40, '*'))
    print(" MAPPING ULTRASUITE PHONEMES TO IPA ... ".center(40, '#'))
    # do mapping
    mapping = []
    with open('path/to/phonemes_to_IPA.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            mapping.append(row)
        print(mapping)
    print(mapping[1][0])

    duration = librosa.get_duration(filename=wav_path)
    segList_IPA = list()
    for i in range(len(segList)):
        start_time = segList[i][0]
        end_time = segList[i][1]
        phoneme = segList[i][2]
        find = 0
        for item in mapping:
            if phoneme == item[0]:
                segList_IPA.append((start_time, end_time, item[1])) #tuple is immutable
                find = 1
        # after the loop, if not find, use the original UltraSuite phonemes
        if find == 0:
            segList_IPA.append((start_time, end_time, phoneme))
    print("segList_IPA", segList_IPA)

    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    phonemeTier = textgrid.IntervalTier('phoneme', segList, 0, segList[-1][1])
    # do mapping here!!!
    phonemeTier_ipa = textgrid.IntervalTier('phoneme_ipa', segList_IPA, 0, segList_IPA[-1][1])
    tg.addTier(phonemeTier)
    tg.addTier(phonemeTier_ipa)
    tg.save(ML_TG_path, format="long_textgrid", includeBlankSpaces=False)
    print(" FINISH PHONETIC GENERATION !!! ".center(40, '*'))



if __name__ == "__main__":
    # # Inference all samples
    # inpath = "/path/to/folder/"
    # for _f in tqdm(os.listdir(inpath)):
    #     parent_f = os.path.join(inpath, _f)
    #     if os.path.isdir(parent_f):
    #         for _ff in os.listdir(parent_f):
    #             parent_ff = os.path.join(parent_f, _ff)
    #             if os.path.isdir(parent_ff):
    #                 for ex in os.listdir(parent_ff):
    #                     if ex.endswith('.wav'): 
    #                         items = ex.split('.')
    #                         temps = items[0].split('_')
    #                         word = temps[-1]
    #                         print(word)
    #                         wav_path = os.path.join(parent_ff, ex)
    #                         print("wav_path", wav_path)
    #                         base_path_1 = os.path.join(os.path.dirname(parent_ff), 'rebase_to_zero_TG')
    #                         if not os.path.exists(base_path_1):
    #                             os.makedirs(base_path_1, exist_ok=True)
    #                         tg_name = ex.replace('.wav', '.TextGrid')
    #                         tg_path = os.path.join(base_path_1, tg_name)
                            
    #                         base_path_2 = os.path.join(os.path.dirname(parent_ff), 'ML_TG')
    #                         if not os.path.exists(base_path_2):
    #                             os.makedirs(base_path_2, exist_ok=True)
    #                         ML_TG_path = os.path.join(base_path_2, tg_name)

    #                         base_path_3 = os.path.join(os.path.dirname(parent_ff), 'individual_json')
    #                         if not os.path.exists(base_path_3):
    #                             os.makedirs(base_path_3, exist_ok=True)
    #                         json_name = ex.replace('.wav', '.json')
    #                         json_path = os.path.join(base_path_3, json_name)
    #                         print("outpath_json",json_path)
    #                         segList = seg_demo(wav_path, json_path)
    #                         generate_textgrid(segList, wav_path, tg_path, ML_TG_path)

    # Inference one
    wav_path = '/path/to/wav/file'
    json_path = "path/to/json/file"
    segList = seg_demo(wav_path,json_path)
    print(segList)
    tg_path = 'path/to/textgrid/file'
    ML_TG_path = 'path/to/save/ML_generated_textgrid'
    generate_textgrid(segList, wav_path, tg_path, ML_TG_path)
