#!/usr/bin/env python
# coding: utf-8

# This program is to compute the correlation between vowel articulation features (automatic and manual) and experts' ratings on speech and voice impairment for TORGO.
# LIU Yuanyuan, 2021-3-23.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import os
import seaborn as sns

def vowel_articulation_expert_rating_corr_TORGO(filepath, task):
    # filepath = '/home/yuanyuan/Documents/TORGO/exp/'
    filepath_target = os.path.join(filepath, 'exp', task)
    df_features = pd.read_excel(filepath_target + 'speaker_formants_stat_auto_None_ipa.xlsx')
    if task == 'TORGO_52':
        df_ratings = pd.read_excel(filepath_target + 'speaker_label_torgo_52audios.xlsx')
    speakers_features = df_features['speaker']
    speakers_ratings = df_ratings['speaker']
    severities = df_ratings['level']
    levels = []
    for i in severities:
        if i == 0:
            levels.append('control')
        elif i == 1:
            levels.append('mild')
        elif i == 2:
            levels.append('moderate')
        elif i == 3:
            levels.append('severe')
    labels = df_features['control']
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
    columns = ['control(mean)', 'control(std)', 'dysarthric(mean)', 'dysarthric(std)', 'diff_ratio', 'ttest', 'pvalue', 'r_speech', 'p_speech']
    df_summary = pd.DataFrame(np.arange(len(groups) * len(his) * 9).reshape(len(groups) * len(his), 9), index=feat_names,
                              columns=columns)

    for feat_name in feat_names:
        feat = df_features[feat_name]
        feat_control = feat.loc[labels == 1]
        feat_dys = feat.loc[labels == 0]
        ttest, p = ttest_ind(feat_control, feat_dys)
        feat_control_mean = round(np.average(feat_control), 2)
        feat_control_std = round(np.std(feat_control), 2)
        feat_dys_mean = round(np.average(feat_dys), 2)
        feat_dys_std = round(np.std(feat_dys), 2)
        diff_ratio = round((feat_control_mean - feat_dys_mean) / feat_dys_mean, 3) * 100
        df_summary.loc[feat_name, 'control(mean)'] = feat_control_mean
        df_summary.loc[feat_name, 'control(std)'] = feat_control_std
        df_summary.loc[feat_name, 'dysarthric(mean)'] = feat_dys_mean
        df_summary.loc[feat_name, 'dysarthric(std)'] = feat_dys_std
        df_summary.loc[feat_name, 'diff_ratio'] = diff_ratio
        df_summary.loc[feat_name, 'ttest'] = round(ttest, 3)
        df_summary.loc[feat_name, 'pvalue'] = round(p, 5)
        if speakers_features.all() == speakers_ratings.all():
            r_speech, p_speech = pearsonr(feat, severities)
            df_summary.loc[feat_name, 'r_speech'] = round(r_speech, 3)
            df_summary.loc[feat_name, 'p_speech'] = round(p_speech, 5)

        data = pd.DataFrame({'severities': severities, feat_name: feat, 'group': levels})
        fig, ax = plt.subplots(1, 1)
        ax = sns.boxplot(x='severities', y=feat_name, data=data, whis=np.inf)
        ax = sns.stripplot(x='severities', y=feat_name, hue='group', data=data, jitter=True, color=".3")
        levels_uni = ['control', 'mild', 'moderate', 'severe']
        x_range = np.arange(len(levels_uni))
        ax.set_xticks(x_range)
        ax.set_xticklabels(levels_uni)
        ax.get_legend().remove()
        ax.set_xlabel('severity level')
        fig.savefig(filepath_target + feat_name + '_scatter_boxplot.pdf')

        fig, ax = plt.subplots()
        ax.set_title(feat_name)
        data = [feat_control, feat_dys]
        ax.boxplot(data, showfliers=False, labels=['control', 'dysarthric'])
        plt.grid(True)
        fig.savefig(filepath_target + feat_name + '_boxplot.png')
    #     plt.close()
    df_summary.to_excel(filepath_target + 'control_dysarthric_summary_pearson.xlsx')
