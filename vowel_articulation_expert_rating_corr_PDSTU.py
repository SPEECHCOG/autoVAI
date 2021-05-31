#!/usr/bin/env python
# coding: utf-8

# This program is to compute the correlation between vowel articulation features (automatic and manual) and experts' ratings on speech and voice impairment for PDSTU.
# LIU Yuanyuan, 2021-3-23.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.ticker as ticker

def vowel_articulation_expert_rating_corr_PDSTU(filepath, set_a, task):
    # filepath = '/home/yuanyuan/Documents/VAI_data_2020/exp/PDSTU_read/'
    df_features_auto = pd.read_excel(os.path.join(filepath, 'speaker_formants_stat_auto_None_ipa.xlsx'))
    if set_a == ['AA']:
        df_features_auto = pd.read_excel(os.path.join(filepath, 'speaker_formants_stat_auto_None_ipa_small.xlsx'))
    df_features_man = pd.read_excel(os.path.join(filepath, 'speaker_formants_stat_man_None.xlsx'))
    if task == 'PDSTU_read':
        df_ratings = pd.read_excel('/home/yuanyuan/Documents/VAI_data_2020/Expert_Rater_Data.xlsx')
    feat_groups = ['VAI', 'VSA', 'FCR', 'F2IU']
    his = ['', '50', '70', '90']
    feat_names = []
    for feat_group in feat_groups:
        for hi in his:
            if hi == '':
                feat_name = feat_group
            else:
                feat_name = feat_group + '[' + hi + ']'

            feat_names.append(feat_name)
    # print(feat_names)
    columns = ['r_speech', 'p_speech', 'r_voice', 'p_voice', 'r_overall', 'p_overall', 'tstat', 'p_ttest']
    df_summary_auto = pd.DataFrame(np.arange(len(feat_groups) * len(his) * 8).reshape(len(feat_groups) * len(his), 8),
                                   index=feat_names, columns=columns)
    df_summary_man = pd.DataFrame(np.arange(len(feat_groups) * len(his) * 8).reshape(len(feat_groups) * len(his), 8),
                                  index=feat_names, columns=columns)

    keys = ['speech','voice', 'overall']
    ratings = {'intelligibility': 'Intelligibility_Avg.', 'voice': 'Voice_Avg.', 'overall': 'Overall_Avg.'}

    speakers_auto = df_features_auto['speaker']
    speakers_man = df_features_man['speaker']
    speakers_ratings = df_ratings['speaker']
    severities = df_ratings['severity']
    group = []
    idx_pd = []
    idx_control = []
    for i in range(len(severities)):
        if severities[i] == 0:
            group.append('control')
            idx_control.append(i)
        else:
            group.append('PD')
            idx_pd.append(i)
    idx_spkrated_auto = []
    idx_spkrated_man = []
    for speaker in speakers_ratings:
        idx_spkrated_auto.append(list(speakers_auto).index(speaker))
        idx_spkrated_man.append(list(speakers_man).index(speaker))

    spkrated_auto = speakers_auto[idx_spkrated_auto]
    spkrated_man = speakers_man[idx_spkrated_man]
    df_features_auto_rated = df_features_auto.iloc[idx_spkrated_auto]
    df_features_man_rated = df_features_man.iloc[idx_spkrated_man]

    if (spkrated_auto.all() == speakers_ratings.all()) and (spkrated_man.all() == speakers_ratings.all()):

        for feat in feat_names:
            y_auto = df_features_auto_rated[feat]
            y_man = df_features_man_rated[feat]
            y_auto_PD = (np.array(y_auto))[idx_pd]
            y_auto_control = (np.array(y_auto))[idx_control]
            y_man_PD = (np.array(y_man))[idx_pd]
            y_man_control = (np.array(y_man))[idx_control]
            tstat_auto, ptt_auto = ttest_ind(y_auto_PD, y_auto_control)
            tstat_man, ptt_man = ttest_ind(y_man_PD, y_man_control)
            if ptt_auto <= 0.001:
                pttstar_auto = '***'
            elif ptt_auto <= 0.01:
                pttstar_auto = '**'
            elif ptt_auto <= 0.05:
                pttstar_auto = '*'
            else:
                pttstar_auto = ''
            df_summary_auto.loc[feat, 'tstat'] = round(tstat_auto, 3)
            df_summary_auto.loc[feat, 'p_ttest'] = round(ptt_auto, 5)
            tstat_auto = round(tstat_auto, 2)
            if ptt_man <= 0.001:
                pttstar_man = '***'
            elif ptt_man <= 0.01:
                pttstar_man = '**'
            elif ptt_man <= 0.05:
                pttstar_man = '*'
            else:
                pttstar_man = ''
            df_summary_man.loc[feat, 'tstat'] = round(tstat_man, 3)
            df_summary_man.loc[feat, 'p_ttest'] = round(ptt_man, 5)
            tstat_man = round(tstat_man, 2)

            y_min = round(np.min(np.array([y_auto, y_man])), 1)
            y_max = round(np.max(np.array([y_auto, y_man])), 1)
            r = []
            pstar = []
            for key in keys:
                rating = ratings[key]
                #             print(feat, rating)
                x = df_ratings[rating]
                r_auto, p_auto = pearsonr(y_auto, x)
                m_auto, b_auto = np.polyfit(x, y_auto, 1)
                if p_auto <= 0.001:
                    pstar_auto = '***'
                elif p_auto <= 0.01:
                    pstar_auto = '**'
                elif p_auto <= 0.05:
                    pstar_auto = '*'
                else:
                    pstar_auto = ''

                df_summary_auto.loc[feat, 'r_' + key] = round(r_auto, 3)
                df_summary_auto.loc[feat, 'p_' + key] = round(p_auto, 5)
                r_auto = round(r_auto, 2)
                data_auto = pd.DataFrame({ratings[key]: np.around(np.array(x), 1), \
                                          feat: np.array(y_auto), 'group': group, 'method': ['automatic'] * len(group)})

                r_man, p_man = pearsonr(y_man, x)
                m_man, b_man = np.polyfit(x, y_man, 1)

                if p_man <= 0.001:
                    pstar_man = '***'
                elif p_man <= 0.01:
                    pstar_man = '**'
                elif p_man <= 0.05:
                    pstar_man = '*'
                else:
                    pstar_man = ''
                df_summary_man.loc[feat, 'r_' + key] = round(r_man, 3)
                df_summary_man.loc[feat, 'p_' + key] = round(p_man, 5)
                r_man = round(r_man, 2)
                data_man = pd.DataFrame({ratings[key]: np.around(np.array(x), 1), \
                                         feat: np.array(y_man), 'group': group, 'method': ['manual'] * len(group)})
                r.append([r_auto, r_man])
                pstar.append([pstar_auto, pstar_man])

                # scatter plot for both auto and man data.
                data_auto_man = pd.concat([data_auto, data_man])
                fig, ax = plt.subplots(1, 1)
                sns.scatterplot(x=ratings[key], y=feat, hue=data_auto_man.group.tolist(), \
                                style=data_auto_man.method.tolist(), data=data_auto_man, \
                                palette=['r', 'b'], s=40)
                x_new = np.arange(round(min(x)), round(max(x)), 5)
                ax.plot(x_new, m_auto * x_new + b_auto, color='black', linestyle='solid')  # auto
                ax.plot(x_new, m_man * x_new + b_man, color='k', linestyle='dotted')  # manual
                if key == 'overall':
                    print('yes set xticks')
                    ax.set_xlim(0, 68)
                    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 71, 10)))
                    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0, 71, 2)))
                    ax.set_xlabel('overall severity')
                ax.set_ylim(y_min, y_max)

                ax.set_ylabel(feat)
                legend = ax.legend(loc='upper right')
                fig.savefig(filepath + 'scatter_plot_' + feat + '_' + key + '_auto_man_expert.pdf')

            #         print('{} & ${}^{}$ & ${}^{}$ & ${}^{}$ & ${}^{}$ & ${}^{}$ & ${}^{}$ & ${}^{}$ & ${}^{}$ \\\\'.format(feat, r[0][0], pstar[0][0],\
    #                                                                              r[1][0], pstar[1][0], r[2][0], pstar[2][0], tstat_auto, pttstar_auto,\
    #                                                                              r[0][1],pstar[0][1], r[1][1],pstar[1][1],r[2][1],pstar[2][1], tstat_man, pttstar_man))
    if set_a == ['AA']:
        df_summary_auto.to_excel(os.path.join(filepath, 'assessment_auto_pearson_ipa_None_expert_small.xlsx'))
    else:
        df_summary_auto.to_excel(os.path.join(filepath, 'assessment_auto_pearson_ipa_None_expert.xlsx'))

    df_summary_man.to_excel(os.path.join(filepath, 'assessment_man_pearson_None_expert.xlsx'))