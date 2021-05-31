# This program is to detect obvious errors of frame-level formants detection (for voiced frames or corner vowel related frames).
# LIU Yuanyuan, TUT, 2020-6-11.

# Reference paper: Acoustic characteristics of American English vowels, James Hillenbrand, Laura A. Getty, Michael J. Clark, and Kimberlee Wheeler.

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def formant_errors(filepath_source, task, am):

    F1a_modal = 1002
    F2a_modal = 1688
    F1i_modal = 452
    F2i_modal = 3081
    F1u_modal = 494
    F2u_modal = 1345
    F1_up = 1000
    F2_up = 3000
    # F1_up = 800
    # F2_up = 2400
    sigma = 0.05

    # for 'ipa' am, each speaker has one set of frame ids related to each of the 3 corner vowels.


    # define a line with slope and bias
    k = (F1i_modal - F1a_modal)/(F2i_modal - F2a_modal)
    b = F1i_modal - k*F2i_modal

    # for task in tasks:
    # print(task)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('F2/Hz')
    ax.set_ylabel('F1/Hz')
    # ax.set_title('Formants error detection for task of {:s}'.format(task))
    # ax.plot([0, F1_up*(1+sigma)], [F2_up*(1+sigma), 0])
    # ax.plot([0, F1_up], [F2_up, F2_up], color='g')
    ax.plot([F2a_modal, 3400], [F1_up, F1_up], color='r')
    ax.plot([F2a_modal, F2a_modal], [F1a_modal, 2100], color='r')
    # ax.plot([0, F1i_modal], [0, F2i_modal], color='g')
    ax.plot([F2a_modal, F2i_modal], [k*F2a_modal+b, k*F2i_modal+b], 'k')
    ax.plot([F2a_modal, F2u_modal], [F1a_modal, F1u_modal], 'k')
    ax.plot([F2u_modal, F2i_modal], [F1u_modal, F1i_modal], 'k')
    ax.plot([F2i_modal], [F1i_modal], 'g^')
    ax.plot([F2a_modal], [F1a_modal], 'ro')
    ax.plot([F2u_modal], [F1u_modal], 'bs')
    ax.text(F2i_modal-100, F1i_modal+100, '/i/ ', color='g')
    ax.text(F2a_modal-100, F1a_modal+50, '/a/ ', color='r')
    ax.text(F2u_modal-100, F1u_modal-100, '/u/ ', color='b')
    ax.axis([1200, 3500, 200, 2200])
    filepath_task = filepath_source + '/exp/'+task + '/'
    filename_txt = os.path.join(filepath_task, 'frame_formants_' + am + '.txt')
    frame_formants = pd.read_csv(filename_txt, sep=',')
    speakers = frame_formants.speaker
    frames = frame_formants.frame
    F1 = frame_formants.F1
    F2 = frame_formants.F2


    filename_candidate_frames = filepath_task + 'candidate_frames_cluster_' + am + '.txt'
    f = open(os.path.join(filepath_task, am+'_formants_errors.txt'), 'w+')
    f.writelines(['speaker,frame,F1,F2\n'])
    row_labels = []
    voiced_F1 = []
    voiced_F2 = []
    voiced_frames = []
    row_num = 0
    dict_errors = {}
    with open(filename_candidate_frames, newline='') as csvfile_voiced_frames:
        voiced_frames_file = csv.reader(csvfile_voiced_frames, delimiter='\t')
        for row in voiced_frames_file:
            row_num += 1
            speaker = row[0]

            all_frames = np.where(np.array(speakers) == speaker)[0]
            all_speakers = np.array(speakers)[all_frames]
            all_F1 = np.array(F1)[all_frames]
            all_F2 = np.array(F2)[all_frames]

            row[1] = row[1] + row[2] + row[3]
            voiced_frames_cur = row[1].replace('[', '').replace(']', '').replace('\'', '').split(',')
            voiced_frames_cur = [int(i) for i in voiced_frames_cur]
            voiced_frames_cur = np.unique(voiced_frames_cur)
            voiced_frames_cur = list(np.sort(voiced_frames_cur))

            for i in range(len(voiced_frames_cur)):
                if voiced_frames_cur[i] >= all_frames.shape[0]:
                    voiced_frames_cur[i] = all_frames.shape[0] - 1
            if row_num == 1:
                row_labels = list(all_speakers[voiced_frames_cur])
                voiced_frames = voiced_frames_cur
                voiced_F1 = list(all_F1[voiced_frames_cur])
                voiced_F2 = list(all_F2[voiced_frames_cur])
                # print('row {:d}'.format(row_num), 'length of voiced_frames {:d}'.format(len(voiced_frames)))
            else:
                # print()
                row_labels = row_labels + list(all_speakers[voiced_frames_cur])
                voiced_frames = voiced_frames + voiced_frames_cur
                voiced_F1 = voiced_F1 + list(all_F1[voiced_frames_cur])
                voiced_F2 = voiced_F2 + list(all_F2[voiced_frames_cur])
                # print('row {:d}'.format(row_num), 'length of voiced_frames {:d}'.format(len(voiced_frames)), 'length of voiced_frames_cur {:d}'.format(len(voiced_frames_cur)))

    num = 0
    num_spk = 0
    speaker_ref = row_labels[0]
    for i in range(len(row_labels)):
        speaker = row_labels[i]
        F1_cur = float(voiced_F1[i])
        F2_cur = float(voiced_F2[i])
        frame_cur = voiced_frames[i] + 1 # to match with frame_formants.txt (frame id starts from 1)
        # if F2_cur > slope * (F1_up*(1+sigma)-F1_cur):
        # if F1_cur > F1_up*(1+sigma) and F2_cur > F2_up*(1+sigma):
        # if F2_cur/F1_cur > F2i_modal/F1i_modal:
        # if F2_cur > F2i_modal*(1+sigma) and F1_cur > F1u_modal*(1+sigma):
        # if (F1_cur > F1i_modal*(1+sigma)) and (F2_cur > (1+sigma)*(k*F1_cur + b)):
        if (F2_cur > F2a_modal * (1 + sigma)) and (F1_cur > (1 + sigma) * F1_up):
            num = num + 1
            # print(speaker, frame_cur, F1_cur, F2_cur)
            ax.plot([F2_cur], [F1_cur], 'o', color='m')
            f.writelines([speaker, ',', str(frame_cur), ',', str(np.around(F1_cur, 1)), ',', str(np.around(F2_cur, 1)), '\n'])
        if speaker != speaker_ref:
            # print(speaker_ref, num)
            dict_errors[speaker_ref] = [num, round(num/num_spk, 3)]
            num = 0
            num_spk = 0
            speaker_ref = speaker
        else:
            num_spk = num_spk + 1
        if i == len(row_labels)-1:
            # print(speaker, num)
            dict_errors[speaker] = [num, round(num/num_spk, 3)]
    # print(num)
    print(pd.DataFrame(list(dict_errors.items())))
    errors_df = pd.DataFrame(dict_errors).T
    errors_df.columns = ['number', 'ratio']
    errors_df.to_excel(filepath_task + am + '_formants_errors_summary.xlsx')
    fig.savefig(os.path.join(filepath_task, am + '_formants_errors.pdf'))
    f.close()
    plt.show()
    plt.close()
    # if it keeps showing, f.writelines and fig.savefig cannot execute. So put this line in the end.

