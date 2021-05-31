# This program is to compute the mean of F1 and F2 for frame clusters of /a/, /i/, and /u/ for each speaker respectively.
# LIU YUANYUAN, TUT, 2020-04-23.

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

def vowel_articulation_auto(filepath_source, task, am, set_a):
    # start of functions definition.
    ## formants to compute VAI and VSA are averaged values.
    def VAI_compute(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u):
        VAI = (f2_i + f1_a) / (f1_i + f1_u + f2_u + f2_a)
        VAI = round(VAI, 4)
        return VAI
    def VSA_compute(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u):
        VSA = 0.5 * ((f2_u + f2_i) * (f1_u - f1_i) - (f2_a + f2_u) * (f1_a - f1_u) - (f2_a + f2_i) * (f1_a - f1_i))
        VSA = np.abs(round(VSA, 4))
        return VSA

    def FCR_compute(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u):
        FCR = (f1_i + f1_u + f2_u + f2_a) / (f2_i + f1_a)
        FCR = round(FCR, 4)
        return FCR

    def F2IU_compute(f2_i, f2_u):
        F2IU = f2_i / f2_u
        F2IU = round(F2IU, 4)
        return F2IU
    def formants_normalization(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u, method):
        F1_max = max(np.hstack((f1_a, f1_i, f1_u)))
        F1_min = min(np.hstack((f1_a, f1_i, f1_u)))
        F2_max = max(np.hstack((f2_a, f2_i, f2_u)))
        F2_min = min(np.hstack((f2_a, f2_i, f2_u)))
        F1_mean = np.mean(np.hstack((f1_a, f1_i, f1_u)))
        F1_std = np.std(np.hstack((f1_a, f1_i, f1_u)))
        F2_mean = np.mean(np.hstack((f2_a, f2_i, f2_u)))
        F2_std = np.std(np.hstack((f2_a, f2_i, f2_u)))
        f1_a_mean = np.mean(f1_a)
        f2_a_mean = np.mean(f2_a)
        f1_i_mean = np.mean(f1_i)
        f2_i_mean = np.mean(f2_i)
        f1_u_mean = np.mean(f1_u)
        f2_u_mean = np.mean(f2_u)
        ## a constructed point uu with equal F1 and F2.
        f1_uu_mean = f1_i_mean
        f2_uu_mean = f1_i_mean
        S1 = (f1_a_mean + f1_i_mean + f1_uu_mean) / 3
        S2 = (f2_a_mean + f2_i_mean + f2_uu_mean) / 3
        if method == 'LCE':
            f1_a = f1_a / F1_max
            f2_a = f2_a / F2_max
            f1_i = f1_i / F1_max
            f2_i = f2_i / F2_max
            f1_u = f1_u / F1_max
            f2_u = f2_u / F2_max
        if method == 'Gerstman':
            f1_a = 999 * ((f1_a - F1_min) / (F1_max + F1_min))
            f2_a = 999 * ((f2_a - F2_min) / (F2_max + F2_min))
            f1_i = 999 * ((f1_i - F1_min) / (F1_max + F1_min))
            f2_i = 999 * ((f2_i - F2_min) / (F2_max + F2_min))
            f1_u = 999 * ((f1_u - F1_min) / (F1_max + F1_min))
            f2_u = 999 * ((f2_u - F2_min) / (F2_max + F2_min))
        if method == 'Lobanov':
            f1_a = (f1_a - F1_mean) / F1_std
            f2_a = (f2_a - F2_mean) / F2_std
            f1_i = (f1_i - F1_mean) / F1_std
            f2_i = (f2_i - F2_mean) / F2_std
            f1_u = (f1_u - F1_mean) / F1_std
            f2_u = (f2_u - F2_mean) / F2_std
        if method == 'W&F':
            f1_a = f1_a / S1
            f2_a = f2_a / S2
            f1_i = f1_i / S1
            f2_i = f2_i / S2
            f1_u = f1_u / S1
            f2_u = f2_u / S2
        if method == 'None':
            f1_a = f1_a
            f2_a = f2_a
            f1_i = f1_i
            f2_i = f2_i
            f1_u = f1_u
            f2_u = f2_u
        return f1_a, f2_a, f1_i, f2_i, f1_u, f2_u

    # end of functions definition.

    # filepath_source = os.path.abspath(os.path.join(os.getcwd(), ".."))
    filepath_target = filepath_source + '/exp/' + task
    speakers = []
    frames = []
    F1 = []
    F2 = []

    # reference paper for normalization: Comparing vowel formant normalization methods
    # normalized_methods = ['LCE', 'Gerstman', 'Lobanov', 'W&F', 'None']
    normalized_methods = ['None']

    # to read the frame-level formants file.
    filename_formant = filepath_target + '/frame_formants_' + am + '.txt'
    if os.path.exists(filename_formant):
        print(filename_formant, ' exists.')
    else:
        print('use frame-level formants with hop length of 30ms.')
        filename_formant = filepath_target + '/frame_formants.txt'

    with open(filename_formant, newline='') as csvfile_frame_formants:
        frame_formants_file = csv.reader(csvfile_frame_formants, delimiter=',')
        for row in frame_formants_file:
            # print(row[0], row[1])
            if row[0] != 'speaker':
                speakers.append(row[0])
                frames.append(int(row[1]))
                F1.append(float(row[2]))
                F2.append(float(row[3]))

    speakers_uni = np.unique(speakers)
    df_min_max_formants = pd.DataFrame(np.arange(len(speakers_uni)*4).reshape(len(speakers_uni), 4), index=speakers_uni, columns=['min_F1', 'max_F1', 'min_F2', 'max_F2'])
    for method in normalized_methods:
        print('Formant normalization: ', method)
        formants_stat = []
        speakers_set = []
        if set_a == ['AA']:
            f = open(os.path.join(filepath_target, 'speaker_formants_stat_auto_' + method + '_' + am + '_small.txt'), 'w')
        else:
            f = open(os.path.join(filepath_target, 'speaker_formants_stat_auto_' + method + '_' + am + '.txt'), 'w')
        f.writelines(['speaker\tf1_a\tf2_a\tf1_i\tf2_i\tf1_u\tf2_u\tf1_a[90]\tf2_a[10]\tf1_i[10]\tf2_i[90]\tf1_u[10]\tf2_u[10]\tf1_a[70]\tf2_a[30]\tf1_i[30]\tf2_i[70]\tf1_u[30]\tf2_u[30]\tf1_a[50]\tf2_a[50]\tf1_i[50]\tf2_i[50]\tf1_u[50]\tf2_u[50]\tVAI\tVSA\tFCR\tF2IU\tVAI[90]\tVSA[90]\tFCR[90]\tF2IU[90]\tVAI[70]\tVSA[70]\tFCR[70]\tF2IU[70]\tVAI[50]\tVSA[50]\tFCR[50]\tF2IU[50]\n'])
        # to read the candidate frames of each vowel cluster for each speaker. In 'candidate_frames_cluster.txt', each row represents one speaker.
        num = 0
        if set_a == ['AA']:
            filename_candidates = filepath_target + '/candidate_frames_cluster_' + am + '_small.txt'
        else:
            filename_candidates = filepath_target + '/candidate_frames_cluster_'+am+'.txt'

        with open(filename_candidates, newline='') as csvfile_frames_cluster:
            frames_cluster_file = csv.reader(csvfile_frames_cluster, delimiter='\t')
            for row in frames_cluster_file:
                num = num+1
                speaker = row[0]
                print(speaker)
                speakers_set.append(speaker)
                all_frames = np.where(np.array(speakers) == speaker)[0]
                # print('all_frames:', all_frames)
                all_F1 = np.array(F1)[all_frames]

                # print('size of all_F1', all_F1.shape)
                all_F2 = np.array(F2)[all_frames]

                frames_a = row[1].replace('[', '').replace(']', '').replace('\'', '').split(',')
                frames_a = [int(i) for i in frames_a]
                # print('frames_a:', frames_a)
                for i in range(len(frames_a)):
                    if frames_a[i] >= all_frames.shape[0]:
                        frames_a[i] = all_frames.shape[0] - 1
                # print('frames_a', frames_a)
                frames_i = row[2].replace('[', '').replace(']', '').replace('\'', '').split(',')
                frames_i = [int(i) for i in frames_i]
                for i in range(len(frames_i)):
                    if frames_i[i] >= all_frames.shape[0]:
                        frames_i[i] = all_frames.shape[0] - 1
                # print('max of frames_i', max(frames_i))
                frames_u = row[3].replace('[', '').replace(']', '').replace('\'', '').split(',')
                frames_u = [int(i) for i in frames_u]
                for i in range(len(frames_u)):
                    if frames_u[i] >= all_frames.shape[0]:
                        frames_u[i] = all_frames.shape[0] - 1
                # frames number generated in praat and ipa_recognizer can be 1 frame different.
                f1_a = all_F1[frames_a]
                f2_a = all_F2[frames_a]
                f1_i = all_F1[frames_i]
                f2_i = all_F2[frames_i]
                f1_u = all_F1[frames_u]
                f2_u = all_F2[frames_u]

                f1_vowels = np.append(np.append(f1_a, f1_i), f1_u)
                f2_vowels = np.append(np.append(f2_a, f2_i), f2_u)

                min_F1 = min(f1_vowels)
                max_F1 = max(f1_vowels)
                min_F2 = min(f2_vowels)
                max_F2 = max(f2_vowels)
                if method == 'None':
                    # print('save min and max of formants')
                    df_min_max_formants.loc[speaker, 'min_F1'] = min_F1
                    df_min_max_formants.loc[speaker, 'max_F1'] = max_F1
                    df_min_max_formants.loc[speaker, 'min_F2'] = min_F2
                    df_min_max_formants.loc[speaker, 'max_F2'] = max_F2


                # formant normalization
                [f1_a, f2_a, f1_i, f2_i, f1_u, f2_u] = formants_normalization(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u, method)
                # to plot formants distribution for each speaker, using 70/30 percentiles as corner vowel representatives. (2020-6-26)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(f2_a, f1_a, 'ro', label='/a/')
                ax.plot(f2_i, f1_i, 'g^', label='/i/')
                ax.plot(f2_u, f1_u, 'bs', label='/u/')
                hi = 70
                lo = 100 - hi
                if speaker != 'KK2':
                    ax.plot(np.percentile(f2_a, lo), np.percentile(f1_a, hi), 'ko', markersize=12)
                    ax.plot(np.percentile(f2_i, hi), np.percentile(f1_i, lo), 'k^', markersize=12)
                    ax.plot(np.percentile(f2_u, lo), np.percentile(f1_u, lo), 'ks', markersize=12)
                legend = ax.legend(loc='upper right')
                ax.set_xlabel('F2')
                ax.set_ylabel('F1')
                if method=='None' and speaker=='K7':
                    ax.set_xlim([600, 3000])
                    ax.set_ylim([200, 1300])
                    ax.set_xlabel('F2/Hz')
                    ax.set_ylabel('F1/Hz')
                    plt.grid(True)
                if method=='None' and speaker=='KK2':
                    ax.set_xlim([600, 3300])
                    ax.set_ylim([100, 2200])
                    ax.set_xlabel('F2/Hz')
                    ax.set_ylabel('F1/Hz')
                    plt.grid(True)
                # plt.show()
                if set_a == ['AA']:
                    fig.savefig(os.path.join(filepath_target, speaker + '_vowels_' + method + '_' + am + '_small.pdf'))
                else:
                    fig.savefig(os.path.join(filepath_target, speaker + '_vowels_' + method + '_' + am + '.pdf'))
                plt.close()

                # to compute vowel articulation features using means.
                f1_a_mean = np.mean(f1_a)
                f2_a_mean = np.mean(f2_a)
                f1_i_mean = np.mean(f1_i)
                f2_i_mean = np.mean(f2_i)
                f1_u_mean = np.mean(f1_u)
                f2_u_mean = np.mean(f2_u)
                VAI = VAI_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
                VSA = VSA_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
                FCR = FCR_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
                F2IU = F2IU_compute(f2_i_mean, f2_u_mean)

            # to compute vowel articulation features using percentiles.
                VAI_prc = None
                VSA_prc = None
                FCR_prc = None
                F2IU_prc = None
                f1_a_prc = None
                f2_a_prc = None
                f1_i_prc = None
                f2_i_prc = None
                f1_u_prc = None
                f2_u_prc = None
                for hi in [90, 70, 50]:
                    lo = 100-hi
                    f1_a_hi = np.percentile(f1_a, hi)
                    f2_a_lo = np.percentile(f2_a, lo)
                    f1_i_lo = np.percentile(f1_i, lo)
                    f2_i_hi = np.percentile(f2_i, hi)
                    f1_u_lo = np.percentile(f1_u, lo)
                    f2_u_lo = np.percentile(f2_u, lo)
                    VAI_prc_cur = VAI_compute(f1_a_hi, f2_a_lo, f1_i_lo, f2_i_hi, f1_u_lo, f2_u_lo)
                    VSA_prc_cur = VSA_compute(f1_a_hi, f2_a_lo, f1_i_lo, f2_i_hi, f1_u_lo, f2_u_lo)
                    FCR_prc_cur = FCR_compute(f1_a_hi, f2_a_lo, f1_i_lo, f2_i_hi, f1_u_lo, f2_u_lo)
                    F2IU_prc_cur = F2IU_compute(f2_i_hi, f2_u_lo)
                    if VAI_prc is None:
                        VAI_prc = VAI_prc_cur
                        VSA_prc = VSA_prc_cur
                        FCR_prc = FCR_prc_cur
                        F2IU_prc = F2IU_prc_cur
                        f1_a_prc = f1_a_hi
                        f2_a_prc = f2_a_lo
                        f1_i_prc = f1_i_lo
                        f2_i_prc = f2_i_hi
                        f1_u_prc = f1_u_lo
                        f2_u_prc = f2_u_lo
                    else:
                        VAI_prc = np.hstack((VAI_prc, VAI_prc_cur))
                        VSA_prc = np.hstack((VSA_prc, VSA_prc_cur))
                        FCR_prc = np.hstack((FCR_prc, FCR_prc_cur))
                        F2IU_prc = np.hstack((F2IU_prc, F2IU_prc_cur))
                        f1_a_prc = np.hstack((f1_a_prc, f1_a_hi))
                        f2_a_prc = np.hstack((f2_a_prc, f2_a_lo))
                        f1_i_prc = np.hstack((f1_i_prc, f1_i_lo))
                        f2_i_prc = np.hstack((f2_i_prc, f2_i_hi))
                        f1_u_prc = np.hstack((f1_u_prc, f1_u_lo))
                        f2_u_prc = np.hstack((f2_u_prc, f2_u_lo))

                formants_stat.append([f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean, f1_a_prc[0], f2_a_prc[0], f1_i_prc[0], f2_i_prc[0], f1_u_prc[0], f2_u_prc[0]])
                f.writelines([speaker, '\t', str(round(f1_a_mean, 2)), '\t', str(round(f2_a_mean, 2)), '\t', str(round(f1_i_mean, 2)), '\t', str(round(f2_i_mean, 2)), '\t', str(round(f1_u_mean, 2)), '\t', str(round(f2_u_mean, 2)), '\t'])
                f.writelines([str(round(f1_a_prc[0], 2)), '\t', str(round(f2_a_prc[0], 2)), '\t', str(round(f1_i_prc[0], 2)), '\t', str(round(f2_i_prc[0], 2)), '\t', str(round(f1_u_prc[0], 2)), '\t', str(round(f2_u_prc[0], 2)), '\t'])
                f.writelines([str(round(f1_a_prc[1], 2)), '\t', str(round(f2_a_prc[1], 2)), '\t', str(round(f1_i_prc[1], 2)), '\t', str(round(f2_i_prc[1], 2)), '\t', str(round(f1_u_prc[1], 2)), '\t', str(round(f2_u_prc[1], 2)), '\t'])
                f.writelines([str(round(f1_a_prc[2], 2)), '\t', str(round(f2_a_prc[2], 2)), '\t', str(round(f1_i_prc[2], 2)), '\t', str(round(f2_i_prc[2], 2)), '\t', str(round(f1_u_prc[2], 2)), '\t', str(round(f2_u_prc[2], 2)), '\t'])

                f.writelines([str(VAI), '\t', str(VSA), '\t', str(FCR), '\t', str(F2IU), '\t'])
                f.writelines([str(VAI_prc[0]), '\t', str(VSA_prc[0]), '\t', str(FCR_prc[0]), '\t', str(F2IU_prc[0]), '\t'])
                f.writelines([str(VAI_prc[1]), '\t', str(VSA_prc[1]), '\t', str(FCR_prc[1]), '\t', str(F2IU_prc[1]), '\t'])
                f.writelines([str(VAI_prc[2]), '\t', str(VSA_prc[2]), '\t', str(FCR_prc[2]), '\t', str(F2IU_prc[2]), '\n'])

        print('Number of speakers:', num)
        formants_stat = np.array(formants_stat)

        # df_min_max_formants.to_excel(filepath_target + '/speaker_vowels_min_max_formants_auto.xlsx')

        # # plot figures
        # fig, ax = plt.subplots(1, 2)
        # fig.suptitle('Automatic computation (formant normalization: ' + method + ')')
        # ax[0].plot(formants_stat[:, 1], formants_stat[:, 0], 'ro', label='/a/')
        # ax[0].plot(formants_stat[:, 3], formants_stat[:, 2], 'g^', label='/i/')
        # ax[0].plot(formants_stat[:, 5], formants_stat[:, 4], 'bs', label='/u/')
        # legend = ax[0].legend(loc='upper right')
        # ax[0].set(xlabel='F2', ylabel='F1')
        # ax[0].set_title('Mean formants')
        # # ax[0].set_xlim(700, 2700)
        # # ax[0].set_ylim(250, 850)
        # ax[1].plot(formants_stat[:, 7], formants_stat[:, 6], 'ro', label='/a/')
        # ax[1].plot(formants_stat[:, 9], formants_stat[:, 8], 'g^', label='/i/')
        # ax[1].plot(formants_stat[:, 11], formants_stat[:, 10], 'bs', label='/u/')
        # ax[1].set_title('Apices of F1 and F2')
        # # legend = ax[1].legend(loc='upper right')
        # ax[1].set(xlabel='F2')
        # # ax[1].set_xlim(800, 2800)
        # # ax[1].set_ylim(150, 450)
        # # plt.show()
        if set_a == ['AA']:
            fig.savefig(os.path.join(filepath_target, 'Formants_vowel_auto_' + method + '_' + am + '_small.pdf'))
        else:
            fig.savefig(os.path.join(filepath_target, 'Formants_vowel_auto_' + method + '_' + am + '.pdf'))
        # plt.close()
