#!/usr/bin/env python
# coding: utf-8

# This program is to compute the correlation between vowel articulation features (automatic and manual) and experts' ratings on speech and voice impairment for PC-GITA read speech.
# LIU Yuanyuan, 2021-3-23.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import ttest_ind
import seaborn as sns
import os

def vowel_articulation_expert_rating_corr_PCGITA(filepath, task):
    # filepath = '/home/yuanyuan/Documents/VAI_data_2020/exp/PC-GITA_read/'
    # feature file and rating file should sort according to speaker first.
    filepath_task = os.path.join(filepath, 'exp', task)
    df_features = pd.read_excel(os.path.join(filepath_task, 'speaker_formants_stat_auto_None_ipa.xlsx'))
    if task == 'PC-GITA_read':
        df_ratings = pd.read_excel(os.path.join(filepath_task, 'PCGITA_metadata.xlsx'))
    UPDRS = df_ratings['UPDRS']
    PD_labels = np.zeros(np.array(UPDRS).shape)
    PD_labels[UPDRS != 0] = 1
    speech = df_ratings['UPDRS-speech']
    speakers_rating = df_ratings['NAME']
    speakers_feature = df_features['speaker']
    groups = ['VAI', 'VSA', 'FCR', 'F2IU']
    his = ['', '50', '70', '90']
    feat_names = []
    for group in groups:
        for hi in his:
            if hi == '':
                feat_name = group
            else:
                feat_name = group + '[' + hi + ']'
            feat_names.append(feat_name)
    # print(feat_names)
    columns = ['control(mean)', 'control(std)', 'PD(mean)', 'PD(std)', 'diff_ratio', 'ttest', 'pvalue']
    df_summary = pd.DataFrame(np.arange(len(groups) * len(his) * 7).reshape(len(groups) * len(his), 7), index=feat_names, columns=columns)
    df_corr = pd.DataFrame(np.arange(len(feat_names) * 4).reshape(len(feat_names), 4), index=feat_names,
                           columns=['r_updrs', 'p_updrs', 'r_speech', 'p_speech'])
    if speakers_rating.all() == speakers_feature.all():
        print('equal speaker order.')
        for feat_name in feat_names:
            feat = df_features[feat_name]
            feat_control = feat.loc[UPDRS == 0]
            feat_dys = feat.loc[UPDRS != 0]
            ttest, p = ttest_ind(feat_control, feat_dys)
            feat_control_mean = round(np.average(feat_control), 2)
            feat_control_std = round(np.std(feat_control), 2)
            feat_dys_mean = round(np.average(feat_dys), 2)
            feat_dys_std = round(np.std(feat_dys), 2)
            diff_ratio = round((feat_control_mean - feat_dys_mean) / feat_dys_mean, 3) * 100
            df_summary.loc[feat_name, 'control(mean)'] = feat_control_mean
            df_summary.loc[feat_name, 'control(std)'] = feat_control_std
            df_summary.loc[feat_name, 'PD(mean)'] = feat_dys_mean
            df_summary.loc[feat_name, 'PD(std)'] = feat_dys_std
            df_summary.loc[feat_name, 'diff_ratio'] = diff_ratio
            df_summary.loc[feat_name, 'ttest'] = round(ttest, 3)
            df_summary.loc[feat_name, 'pvalue'] = round(p, 5)
            data = pd.DataFrame({'PD_labels': PD_labels, feat_name: feat})

            fig, ax = plt.subplots(1, 1)
            ax = sns.stripplot(x='PD_labels', y=feat_name, data=data, jitter=True, color='.3')
            ax = sns.boxplot(x='PD_labels', y=feat_name, data=data, whis=np.inf)
            levels_uni = ['control', 'PD']
            x_range = np.arange(len(levels_uni))
            ax.set_xticks(x_range)
            ax.set_xticklabels(levels_uni)
            ax.set_xlabel('')
            #         ax.get_legend().remove()
            fig.savefig(filepath_task + feat_name + '_scatter_boxplot.pdf')


            # scatter plot with UPDRS (and speech) ratings
            #         fig, ax = plt.subplots(1, 1)
            x = UPDRS
            y_auto = feat
            r, p = scipy.stats.pearsonr(y_auto, x)
            df_corr.loc[feat_name, 'r_updrs'] = round(r, 3)
            df_corr.loc[feat_name, 'p_updrs'] = round(p, 5)
            #         ax.scatter(x[UPDRS == 0], y_auto[UPDRS == 0], c='b', label='control')
            #         ax.scatter(x[UPDRS != 0], y_auto[UPDRS != 0], c='r', label='PD')
            #         legend = ax.legend(loc='upper right')
            #         ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
            #         ax.set_xlabel('UPDRS')
            #         ax.set_ylabel(feat_name+' (automatic)')
            #         fig.savefig(filepath_task+'scatter_plot_'+feat_name+'_UPDRS_auto.pdf')
            #         plt.close()

            # scatter plot with speech ratings
            #         fig, ax = plt.subplots(1, 1)
            x = speech
            y_auto = feat
            r, p = scipy.stats.pearsonr(y_auto, x)
            df_corr.loc[feat_name, 'r_speech'] = round(r, 3)
            df_corr.loc[feat_name, 'p_speech'] = round(p, 5)
        #         ax.scatter(x[UPDRS == 0], y_auto[UPDRS == 0], c='b', label='control')
        #         ax.scatter(x[UPDRS != 0], y_auto[UPDRS != 0], c='r', label='PD')
        #         legend = ax.legend(loc='upper right')
        #         ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        #         ax.set_xlabel('UPDRS-speech')
        #         ax.set_ylabel(feat_name+' (automatic)')
        #         fig.savefig(filepath_task+'scatter_plot_'+feat_name+'_speech_auto.pdf')
        #         plt.close()

        df_corr.to_excel(filepath_task + 'corr_vowel_articulation_ratings.xlsx')
        df_summary.to_excel(filepath_task + 'control_PD_summary.xlsx')

        ## formants distribution
        for hi in his:
            if hi == '':
                F1a = df_features['f1_a']
                F2a = df_features['f2_a']
                F1i = df_features['f1_i']
                F2i = df_features['f2_i']
                F1u = df_features['f1_u']
                F2u = df_features['f2_u']
            else:
                lo = str(100 - int(hi))
                F1a = df_features['f1_a[' + hi + ']']
                F2a = df_features['f2_a[' + lo + ']']
                F1i = df_features['f1_i[' + lo + ']']
                F2i = df_features['f2_i[' + hi + ']']
                F1u = df_features['f1_u[' + lo + ']']
                F2u = df_features['f2_u[' + lo + ']']
            F1_min = np.floor(np.min(F1u) / 100) * 100
            F1_max = np.ceil(np.max(F1a) / 100) * 100
            F2_min = np.floor(np.min(F2u) / 100) * 100
            F2_max = np.ceil(np.max(F2i) / 100) * 100

            #     fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            fig, axs = plt.subplots()
            if hi == '':
                print('Control: Formants distribution (mean of frame-level formants)')
            else:
                print('Control: Formants distribution (percentiles: ' + hi + '/' + str(100 - int(hi)) + ')')
            axs.plot(F2a.loc[UPDRS == 0], F1a.loc[UPDRS == 0], 'ro', label='/a/')
            axs.plot(F2i.loc[UPDRS == 0], F1i.loc[UPDRS == 0], 'g^', label='/i/')
            axs.plot(F2u.loc[UPDRS == 0], F1u.loc[UPDRS == 0], 'bs', label='/u/')
            axs.plot(np.average(F2a.loc[UPDRS == 0]), np.average(F1a.loc[UPDRS == 0]), 'ko', markersize=12)
            axs.plot(np.average(F2i.loc[UPDRS == 0]), np.average(F1i.loc[UPDRS == 0]), 'k^', markersize=12)
            axs.plot(np.average(F2u.loc[UPDRS == 0]), np.average(F1u.loc[UPDRS == 0]), 'ks', markersize=12)
            legend = axs.legend(loc='upper right')
            axs.set(xlabel='F2/Hz', ylabel='F1/Hz')
            #     axs[0].set_title('Control')
            axs.set_xlim(F2_min, F2_max)
            axs.set_ylim(F1_min, F1_max)
            plt.grid(True)
            if hi == '':
                fig.savefig(filepath_task + 'Formants_vowel_auto_None_ipa_mean_formants_control.pdf')
            else:
                fig.savefig(filepath_task + 'Formants_vowel_auto_None_ipa_' + hi + '_' + str(100 - int(hi)) + '_control.pdf')
            plt.close()

            fig, axs = plt.subplots()
            if hi == '':
                print('PD: Formants distribution (mean of frame-level formants)')
            else:
                print('PD: Formants distribution (percentiles: ' + hi + '/' + str(100 - int(hi)) + ')')
            axs.plot(F2a.loc[UPDRS != 0], F1a.loc[UPDRS != 0], 'ro', label='/a/')
            axs.plot(F2i.loc[UPDRS != 0], F1i.loc[UPDRS != 0], 'g^', label='/i/')
            axs.plot(F2u.loc[UPDRS != 0], F1u.loc[UPDRS != 0], 'bs', label='/u/')
            axs.plot(np.average(F2a.loc[UPDRS != 0]), np.average(F1a.loc[UPDRS != 0]), 'ko', markersize=12)
            axs.plot(np.average(F2i.loc[UPDRS != 0]), np.average(F1i.loc[UPDRS != 0]), 'k^', markersize=12)
            axs.plot(np.average(F2u.loc[UPDRS != 0]), np.average(F1u.loc[UPDRS != 0]), 'ks', markersize=12)
            legend = axs.legend(loc='upper right')
            axs.set(xlabel='F2/Hz', ylabel='F1/Hz')
            #     axs[0].set_title('Dysarthric')
            axs.set_xlim(F2_min, F2_max)
            axs.set_ylim(F1_min, F1_max)
            plt.grid(True)
            if hi == '':
                fig.savefig(filepath_task + 'Formants_vowel_auto_None_ipa_mean_formants_PD.pdf')
            else:
                fig.savefig(filepath_task + 'Formants_vowel_auto_None_ipa_' + hi + '_' + str(100 - int(hi)) + '_PD.pdf')
            plt.close()