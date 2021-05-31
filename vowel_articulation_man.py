# This program is to compute mean and percentiles of f1 and f2 for corner vowels with manual annotations.
# LIU YUANYUAN, TUT, 2020-4-27.

import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def vowel_articulation_man(filepath_source, task):
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
        f1_max = max(np.hstack((f1_a, f1_i, f1_u)))
        f1_min = min(np.hstack((f1_a, f1_i, f1_u)))
        f2_max = max(np.hstack((f2_a, f2_i, f2_u)))
        f2_min = min(np.hstack((f2_a, f2_i, f2_u)))
        f1_mean = np.mean(np.hstack((f1_a, f1_i, f1_u)))
        f1_std = np.std(np.hstack((f1_a, f1_i, f1_u)))
        f2_mean = np.mean(np.hstack((f2_a, f2_i, f2_u)))
        f2_std = np.std(np.hstack((f2_a, f2_i, f2_u)))
        f1_a_mean = np.mean(f1_a)
        f2_a_mean = np.mean(f2_a)
        f1_i_mean = np.mean(f1_i)
        f2_i_mean = np.mean(f2_i)
        f1_u_mean = np.mean(f1_u)
        f2_u_mean = np.mean(f2_u)
        ## a constructed point uu with equal f1 and f2.
        f1_uu_mean = f1_i_mean
        f2_uu_mean = f1_i_mean
        S1 = (f1_a_mean + f1_i_mean + f1_uu_mean) / 3
        S2 = (f2_a_mean + f2_i_mean + f2_uu_mean) / 3
        if method == 'LCE':
            f1_a = f1_a / f1_max
            f2_a = f2_a / f2_max
            f1_i = f1_i / f1_max
            f2_i = f2_i / f2_max
            f1_u = f1_u / f1_max
            f2_u = f2_u / f2_max
        if method == 'Gerstman':
            f1_a = 999 * ((f1_a - f1_min) / (f1_max + f1_min))
            f2_a = 999 * ((f2_a - f2_min) / (f2_max + f2_min))
            f1_i = 999 * ((f1_i - f1_min) / (f1_max + f1_min))
            f2_i = 999 * ((f2_i - f2_min) / (f2_max + f2_min))
            f1_u = 999 * ((f1_u - f1_min) / (f1_max + f1_min))
            f2_u = 999 * ((f2_u - f2_min) / (f2_max + f2_min))
        if method == 'Lobanov':
            f1_a = (f1_a - f1_mean) / f1_std
            f2_a = (f2_a - f2_mean) / f2_std
            f1_i = (f1_i - f1_mean) / f1_std
            f2_i = (f2_i - f2_mean) / f2_std
            f1_u = (f1_u - f1_mean) / f1_std
            f2_u = (f2_u - f2_mean) / f2_std
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
    # end of function definition
    
    # filepath_source = os.path.abspath(os.path.join(os.getcwd(), ".."))
    filepath_target = filepath_source + '/exp/' + task + '/'
    speakers = []
    vowels = []
    f1 = []
    f2 = []
    speakers_set = []
    # reference paper for normalization: Comparing vowel formant normalization methods
    # normalized_methods = ['LCE', 'Gerstman', 'Lobanov', 'W&F', 'None']
    normalized_methods = ['None']
    
    with open(os.path.join(filepath_target, 'vowel_formants_man.txt'), newline='') as csvfile_frame_formants:
        frame_formants_file = csv.reader(csvfile_frame_formants, delimiter=',')
        for row in frame_formants_file:
            if row[0] != 'speaker':
                speakers.append(row[0])
                # print(row)
                vowels.append(row[1])
                f1.append(float(row[4]))
                f2.append(float(row[5]))
    speakers_set = np.unique(np.array(speakers))
    # df_min_max_formants = pd.DataFrame(np.arange(len(speakers_set)*4).reshape(len(speakers_set), 4), index=speakers_set, columns=['min_F1', 'max_F1', 'min_F2', 'max_F2'])
    # Try different formant normalization methods.
    for method in normalized_methods:
        formants_stat = []
        f = open(filepath_target + 'speaker_formants_stat_man_' + method + '.txt', 'w')
        f.writelines(['speaker\tf1_a\tf2_a\tf1_i\tf2_i\tf1_u\tf2_u\tf1_a[90]\tf2_a[10]\tf1_i[10]\tf2_i[90]\tf1_u[10]\tf2_u[10]\tf1_a[70]\tf2_a[30]\tf1_i[30]\tf2_i[70]\tf1_u[30]\tf2_u[30]\tf1_a[50]\tf2_a[50]\tf1_i[50]\tf2_i[50]\tf1_u[50]\tf2_u[50]\tVAI\tVSA\tFCR\tF2IU\tVAI[90]\tVSA[90]\tFCR[90]\tF2IU[90]\tVAI[70]\tVSA[70]\tFCR[70]\tF2IU[70]\tVAI[50]\tVSA[50]\tFCR[50]\tF2IU[50]\n'])
        for speaker in speakers_set:
            rows = np.where(np.array(speakers) == speaker)[0]
            vowels_cur = np.array(vowels)[rows]
            f1_cur = np.array(f1)[rows]
            f2_cur = np.array(f2)[rows]
            rows_a = np.where(np.array(vowels_cur) == 'a')
            rows_i = np.where(np.array(vowels_cur) == 'i')
            rows_u = np.where(np.array(vowels_cur) == 'u')
            f1_a = f1_cur[rows_a]
            f2_a = f2_cur[rows_a]
            f1_i = f1_cur[rows_i]
            f2_i = f2_cur[rows_i]
            f1_u = f1_cur[rows_u]
            f2_u = f2_cur[rows_u]

            f1_vowels = np.append(np.append(f1_a, f1_i), f1_u)
            f2_vowels = np.append(np.append(f2_a, f2_i), f2_u)

            min_F1 = min(f1_vowels)
            max_F1 = max(f1_vowels)
            min_F2 = min(f2_vowels)
            max_F2 = max(f2_vowels)

            # normalization of formants
            [f1_a, f2_a, f1_i, f2_i, f1_u, f2_u] = formants_normalization(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u, method)
            # to plot formants distribution for each speaker. (2020-6-26)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(f2_a, f1_a, 'ro', label='/a/')
            ax.plot(f2_i, f1_i, 'g^', label='/i/')
            ax.plot(f2_u, f1_u, 'bs', label='/u/')
            hi = 70
            lo = 100 - hi
            if speaker != 'KK2':
                ax.plot(np.average(f2_a), np.average(f1_a), 'ko', markersize=12)
                ax.plot(np.average(f2_i), np.average(f1_i), 'k^', markersize=12)
                ax.plot(np.average(f2_u), np.average(f1_u), 'ks', markersize=12)
            legend = ax.legend(loc='upper right')
            ax.set_xlabel('F2')
            ax.set_ylabel('F1')
            if method == 'None' and speaker == 'K7':
                # df_min_max_formants.loc[speaker, 'min_F1'] = min_F1
                # df_min_max_formants.loc[speaker, 'max_F1'] = max_F1
                # df_min_max_formants.loc[speaker, 'min_F2'] = min_F2
                # df_min_max_formants.loc[speaker, 'max_F2'] = max_F2
                ax.set_xlim([600, 3000])
                ax.set_ylim([200, 1300])
                ax.set_xlabel('F2/Hz')
                ax.set_ylabel('F1/Hz')
                plt.grid(True)
            if method == 'None' and speaker == 'KK2':
                ax.set_xlim([600, 3300])
                ax.set_ylim([100, 2200])
                ax.set_xlabel('F2/Hz')
                ax.set_ylabel('F1/Hz')
                plt.grid(True)
            # ax.set_title(speaker + ' (formant normalization:' + method + ')')
            # plt.show()
            fig.savefig(filepath_target + speaker + '_vowels_' + method + '_man.pdf')
            plt.close()
            # to compute VAI and VSA.
            f1_a_mean = np.mean(f1_a)
            f2_a_mean = np.mean(f2_a)
            f1_i_mean = np.mean(f1_i)
            f2_i_mean = np.mean(f2_i)
            f1_u_mean = np.mean(f1_u)
            f2_u_mean = np.mean(f2_u)
            # compute VAI and VSA
            # if method is 'Lobanov', the VAI is constantly -1, because for each speaker, there are 10 occurrences for each of /a/, /i/ and /u/.
            VAI = VAI_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
            VSA = VSA_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
            FCR = FCR_compute(f1_a_mean, f2_a_mean, f1_i_mean, f2_i_mean, f1_u_mean, f2_u_mean)
            F2IU = F2IU_compute(f2_i_mean, f2_u_mean)
            # percentiles of formants
            # to compute VAI and VSA using percentiles.
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
            for prc_hi in [90, 70, 50]:
                prc_lo = 100 - prc_hi
                f1_a_hi = np.percentile(f1_a, prc_hi)
                f2_a_lo = np.percentile(f2_a, prc_lo)
                f1_i_lo = np.percentile(f1_i, prc_lo)
                f2_i_hi = np.percentile(f2_i, prc_hi)
                f1_u_lo = np.percentile(f1_u, prc_lo)
                f2_u_lo = np.percentile(f2_u, prc_lo)
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
        formants_stat = np.array(formants_stat)

        # df_min_max_formants.to_excel(filepath_target+'speaker_vowels_min_max_formants_man.xlsx')
        # # plot
        # fig, ax = plt.subplots(1, 2)
        # fig.suptitle('Computation with manual annotation (formant normalization: ' + method + ')')
        # ax[0].plot(formants_stat[:, 1], formants_stat[:, 0], 'ro', label='/a/')
        # ax[0].plot(formants_stat[:, 3], formants_stat[:, 2], 'g^', label='/i/')
        # ax[0].plot(formants_stat[:, 5], formants_stat[:, 4], 'bs', label='/u/')
        # legend = ax[0].legend(loc='upper right')
        # ax[0].set_title('Mean formants')
        # ax[0].set(xlabel='F2', ylabel='F1')
        #
        # # ax[0].set_xlim(700, 2700)
        # # ax[0].set_ylim(250, 850)
        #
        # ax[1].plot(formants_stat[:, 7], formants_stat[:, 6], 'ro', label='/a/')
        # ax[1].plot(formants_stat[:, 9], formants_stat[:, 8], 'g^', label='/i/')
        # ax[1].plot(formants_stat[:, 11], formants_stat[:, 10], 'bs', label='/u/')
        # # legend = ax[1].legend(loc='upper right')
        # plt.xlabel('F2')
        # ax[1].set_title('Apices of F1 and F2')
        # # ax[1].set(xlabel='f2/ Hz')
        # # ax[1].set_xlim(800, 2800)
        # # ax[1].set_ylim(150, 450)
        # # plt.show()
        # fig.savefig(filepath_source + '/exp/' + task + '/Formants_vowel_man_' + method + '.pdf')
        # plt.close()