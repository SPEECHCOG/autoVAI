# This program is to screen out candidate frames of vowel /a/, /i/ and /u/.
# A frame is selected if: it is recognized as one corner vowel or among the top 4 phonemes with largest posteriors, there is corner vowel with posterior larger than a threshold (2020-07-03). For the recognized phone, the posterior is marked as 1 if the phone is not included in the top 4 phones.
# It also generates a txt file 'candidate_frames_cluster.txt', each row represents one speaker, with column of speaker
# id, frames set for /a/, /i/ and /u/ respectively.
# set_a = ['AA', 'AE', 'AH', 'AW', 'AY']
# set_i = ['IH', 'IX', 'IY']
# set_u = ['UW', 'UH', 'OW']
# LIUYUANYUAN, TUT, 2020-04-22.

import os
# from glob import glob
from ipa_recognizer import Recognizer
import numpy as np
import csv
#
# model = Recognizer()
# phn_eng = model.decoder.unit.id_to_unit
def frames_screen(filepath_source, task, am, posterior_threshold, set_a, set_i, set_u):
    # posterior_threshold = 0.2
    # set_a = ['AA', 'AE', 'AH', 'AW', 'AY']
    # set_i = ['IH', 'IX', 'IY']
    # set_u = ['UW', 'UH', 'OW']
    set_all = set_a + set_i + set_u

    # file_path_cur = os.path.abspath(os.path.join(os.getcwd(), ".."))
    filepath_target = filepath_source + '/exp/' + task

    if set_a == ['AA']:
        f = open(os.path.join(filepath_target, 'candidate_vowel_frames_' + am + '_small.txt'), 'w')
        f_cluster = open(os.path.join(filepath_target, 'candidate_frames_cluster_' + am + '_small.txt'), 'w')
        f_frames = open(os.path.join(filepath_target, 'frames_statistics_' + am + '_small.txt'), 'w')
    else:
        f = open(os.path.join(filepath_target, 'candidate_vowel_frames_' + am + '.txt'), 'w')
        f_cluster = open(os.path.join(filepath_target, 'candidate_frames_cluster_' + am + '.txt'), 'w')
        f_frames = open(os.path.join(filepath_target, 'frames_statistics_' + am + '.txt'), 'w')
    f_frames.writelines(['speaker\ttotal\t<blk>\tblk_ratio\t/a/\t/i/\t/u/\n'])

    with open(os.path.join(filepath_target, am + '_soft_decode.txt'), newline='') as csvfile_phn_post:
        phn_post_file = csv.reader(csvfile_phn_post, delimiter='\t')
        # change_speaker = 0
        frames_A = []
        frames_I = []
        frames_U = []
        frames_blk = 0
        # frames_num = 0
        speaker_cur = 'speaker'
        for row in phn_post_file:
            # speaker_cur = row[0]
            if row[0] != 'speaker':
                speaker = row[0]
                # A new speaker from this row.
                if speaker != speaker_cur:
                    print('A new speaker:', speaker, speaker_cur)
                    if speaker_cur != 'speaker':
                        # print('write file.')
                        f_cluster.writelines([speaker_cur, '\t', str(frames_A), '\t', str(frames_I), '\t', str(frames_U), '\n'])
                        f_frames.writelines(
                            [speaker_cur, '\t', str(frame), '\t', str(frames_blk), '\t', str(round(frames_blk/(float(frame)+1),3)), '\t', str(len(frames_A)), '\t',
                             str(len(frames_I)), '\t', str(len(frames_U)), '\n'])
                    # change_speaker = 1
                    speaker_cur = speaker
                    frames_A = []
                    frames_I = []
                    frames_U = []
                    frames_blk = 0
                    # frames_num += 1
                frame = row[1]
                rec_phn = row[2]
                if rec_phn == '<blk>':
                    frames_blk += 1
                top_phn = np.array(row[4:11:2])
                top_post = row[5:12:2]
                top_post = np.array([float(i) for i in top_post])
                # print(top_phn, top_post)
                phn_sel = top_phn[top_post >= posterior_threshold]
                post_sel = top_post[top_post >= posterior_threshold]
                phn_sel_vowel = [i for i in phn_sel if i in set_all]
                #
                if (rec_phn in set_all) and (rec_phn in phn_sel_vowel)==False:
                    # print(speaker, frame, 'rec_phn:', rec_phn, 'phn_sel_vowel:', phn_sel_vowel)
                    phn_sel_vowel.append(rec_phn)
                    phn_sel = np.hstack((phn_sel, rec_phn))
                    post_sel = np.hstack((post_sel, 1))

                if len(phn_sel_vowel) != 0:
                    f.writelines([speaker, '\t', frame, '\t'])
                    for phn in phn_sel_vowel:
                        post = post_sel[list(phn_sel).index(phn)]
                        f.writelines([phn, '\t', str(post), '\t'])
                        # to throw each candidate frame into one or more clusters.
                        if phn in set_a:
                            frames_A.append(frame)
                        elif phn in set_i:
                            frames_I.append(frame)
                        elif phn in set_u:
                            frames_U.append(frame)
                    f.writelines('\n')
        f_cluster.writelines([speaker, '\t', str(frames_A), '\t', str(frames_I), '\t', str(frames_U), '\n'])
        # print(speaker, 'total frames: ', frame+1, 'recognized <blk> frames: ', frames_blk, 'frames of /a/, /i/, /u/: ', frames_A.shape[0], frames_I.shape[0], frames_U.shape[0])
        f_frames.writelines([speaker, '\t', str(frame), '\t', str(frames_blk), '\t', str(round(frames_blk/(float(frame)+1), 3)), '\t', str(len(frames_A)), '\t', str(len(frames_I)), '\t', str(len(frames_U)), '\n'])
        print('Done!')






