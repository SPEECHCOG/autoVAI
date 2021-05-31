# This program is to study the efficacy of automatic candidate frames screen.
# Input files: exp/candidate_frames_cluster_ipa.txt, data/PDSTU_read/K1_read.txt
# LIU Yuanyuan, TUT, 2020-6-29.

# First convert the annotation file like K1_read.TextGrid to a list K1.txt using data_preparation_part2 in program.

import os
from glob import glob
import pandas as pd
import numpy as np


def manual_frames_cover_ratio(filepath_source, task, am):
    # window_length = 45
    hop_length = 30
    filepath_data = filepath_source + '/data/' + task
    filepath_exp = filepath_source + '/exp/' + task
    f = open(filepath_exp + '/candidate_frames_cluster_man.txt', 'w')
    annotation_files = glob(os.path.join(filepath_data, '*.txt'))
    frames_insertion = dict()
    frames_number = dict()
    for file in annotation_files:
        print(file)
        speaker = os.path.basename(file)
        if task == 'PDSTU_read':
            speaker = speaker[0:speaker.index('_')]
        # print(speaker)
        f.writelines([speaker, '\t'])
        annotation = pd.read_csv(file, sep='\t')
        tmin = annotation['tmin']
        text = annotation['text']
        # print('text: {}'.format(text))
        tmax = annotation['tmax']
        vowels = ['a', 'i', 'u']
        rows = {}
        for vowel in vowels:
            rows[vowel] = []
        for vowel in vowels:
            print('vowel: {}'.format(vowel))
            # rows_short = np.array(text[text.values == vowel].index)
            # rows_long = np.array(text[text.values == vowel + vowel].index)
            rows_short = [i for i in range(len(text)) if text[i] == vowel]
            rows_long = [i for i in range(len(text)) if text[i] == vowel+vowel]
            print('rows_short: {}, length {}'.format(rows_short, len(rows_short)))
            print('rows_long: {}, length {}'.format(rows_long, len(rows_long)))
            # rows[vowel] = np.sort(np.hstack([rows_short, rows_long]))
            rows[vowel] = np.sort(np.array(rows_short+rows_long))
            print('rows[vowel]: {}'.format(rows[vowel]))
            tmin_vowel = tmin[rows[vowel]] * 1000
            tmax_vowel = tmax[rows[vowel]] * 1000
            frame0_vowel = np.floor(tmin_vowel / hop_length).astype(int)
            frame1_vowel = np.floor(tmax_vowel / hop_length).astype(int)
            frames_cur = []
            for row in rows[vowel]:
                frames_cur = np.hstack([frames_cur, np.arange(frame0_vowel[row], frame1_vowel[row] + 1, 1)])
            f.writelines([str(list(frames_cur.astype(int).astype(str)))])
            if vowel != 'u':
                f.writelines('\t')
        f.writelines('\n')
    f.close()

    candidate_man = pd.read_csv(filepath_exp + '/candidate_frames_cluster_man.txt', header=None, sep='\t')
    candidate_man.columns = ['speaker', 'a', 'i', 'u']
    candidate_man = candidate_man.sort_values(by='speaker')
    speakers_man = candidate_man['speaker']

    candidate_am = pd.read_csv(filepath_exp + '/candidate_frames_cluster_' + am + '.txt', header=None, sep='\t')
    candidate_am.columns = ['speaker', 'a', 'i', 'u']
    candidate_am = candidate_am.sort_values(by='speaker')
    speakers_am = candidate_am['speaker']

    for speaker in speakers_man:
        row_man = speakers_man[speakers_man.values == speaker].index
        row_am = speakers_am[speakers_am.values == speaker].index
        ratios = []
        numbers = []
        for vowel in ['a', 'i', 'u']:
            frames_man = str(candidate_man[vowel][row_man].values)
            frames_am = str(candidate_am[vowel][row_am].values)
            frames_man = frames_man.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
            frames_am = frames_am.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
            set_in = [i for i in frames_man if i in frames_am]
            insertion_ratio = np.round(len(set_in) / len(frames_man), 4)
            ratios.append(insertion_ratio)
            numbers.append(len(frames_man))
        frames_insertion[speaker] = ratios
        frames_number[speaker] = numbers
    frames_insertion_df = pd.DataFrame(frames_insertion).T
    frames_insertion_df.columns = ['a', 'i', 'u']
    frames_insertion_df.to_excel(filepath_exp + '/frames_insertion_man_' + am + '.xlsx')
    ax = frames_insertion_df.plot.bar(rot=0)
    frames_number_df = pd.DataFrame(frames_number).T
    frames_number_df.columns = ['a', 'i', 'u']
    frames_number_df.to_excel(filepath_exp + '/frames_number_man.xlsx')


def touched_segments(filepath_source, task, am):
    # window_length = 45
    hop_length = 30
    filepath_data = filepath_source + '/data/' + task
    filepath_exp = filepath_source + '/exp/' + task
    vowels = ['a', 'i', 'u']
    gama = 1 # tolerance of frame number

    f = open(filepath_exp + '/candidate_frames_cluster_man.txt', 'w')
    # annotation_files = glob(os.path.join(filepath_data, '*.txt'))

    candidate_am = pd.read_csv(filepath_exp + '/candidate_frames_cluster_' + am + '.txt', header=None, sep='\t')
    candidate_am.columns = ['speaker', 'a', 'i', 'u']
    candidate_am = candidate_am.sort_values(by='speaker')
    speakers_am = candidate_am['speaker']

    segments_touch = dict()

    for speaker in speakers_am:

        print(speaker)
        touch_num = dict()
        for vowel in vowels:
            touch_num[vowel] = 0
        row_am = speakers_am[speakers_am.values == speaker].index

        if task == 'normal_read':
            annotation = pd.read_csv(os.path.join(filepath_data, speaker + '_read.txt'), sep='\t')
        else:
            annotation = pd.read_csv(os.path.join(filepath_data, speaker + '.txt'), sep='\t')
        tmin = annotation['tmin']
        text = annotation['text']
        tmax = annotation['tmax']
        for i in range(len(text)):
            vowel = text[i]
            vowel = vowel[0]
            t0 = tmin[i] * 1000
            t1 = tmax[i] * 1000
            frame0 = np.floor(t0 / hop_length).astype(int) - gama
            frame1 = np.floor(t1 / hop_length).astype(int) + gama
            frames_man = np.arange(frame0, frame1+1, 1)

            if vowel in vowels:
                frames_am = str(candidate_am[vowel][row_am].values)
                frames_am = frames_am.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').replace('', '').replace(' ', '').split(',')
                num = 0
                for j in frames_man:
                    if str(j) in frames_am:
                        num = num + 1
                    else:
                        pass
                if num > 0:
                    touch_num[vowel] = touch_num[vowel] + 1



        segments_touch[speaker] = [touch_num['a'], touch_num['i'], touch_num['u']]

    segments_touch_df = pd.DataFrame(segments_touch).T
    segments_touch_df.columns = ['a', 'i', 'u']
    segments_touch_df.to_excel(filepath_exp + '/segments_touch_man_' + am + '.xlsx')