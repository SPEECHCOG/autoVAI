# This program is to investigate the correlation between formants parameters generated automatically and manually.
# Input files: speaker_formants_stat_auto.txt, speaker_formants_stat_man.txt, participants_info.xlsx
# LIU YUANYUAN, TUT, 2020-4-29.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from txt2xlsx import txt_to_xlsx

def auto_man_corr(filepath_source, task, am, set_a):
    # filepath_source = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # task = 'read'
    filepath_exp = filepath_source + '/exp/' + task

    normalized_methods = ['None']
    feats = ['VAI', 'VAI[50]', 'VAI[70]', 'VAI[90]', 'VSA', 'VSA[50]', 'VSA[70]',
             'VSA[90]', 'FCR', 'FCR[50]', 'FCR[70]', 'FCR[90]', 'F2IU', 'F2IU[50]', 'F2IU[70]',
             'F2IU[90]']

    # The below two .xlsx files are converted from the regarding .txt files manually.
    # Remember to sort the below two .xlsx files according to 'speaker' first manually.
    for method in normalized_methods:
        print(method)
        if set_a == ['AA']:
            auto_txt = filepath_exp + '/speaker_formants_stat_auto_' + method + '_' + am + '_small.txt'
            auto_xlsx = filepath_exp + '/speaker_formants_stat_auto_' + method + '_' + am + '_small.xlsx'
        else:
            auto_txt = filepath_exp + '/speaker_formants_stat_auto_' + method + '_' + am + '.txt'
            auto_xlsx = filepath_exp + '/speaker_formants_stat_auto_' + method + '_' + am + '.xlsx'
        if os.path.exists(auto_xlsx) == False:
            txt_to_xlsx(auto_txt, auto_xlsx)
        data_auto_ori = pd.read_excel(auto_xlsx, sheet_name='Sheet1')
        data_auto = data_auto_ori.sort_values('speaker')
        data_auto.to_excel(auto_xlsx, index=False)
        data_auto = pd.read_excel(auto_xlsx, sheet_name='Sheet1')
        man_txt = filepath_exp + '/speaker_formants_stat_man_' + method + '.txt'
        man_xlsx = filepath_exp + '/speaker_formants_stat_man_' + method + '.xlsx'
        if os.path.exists(man_xlsx) == False:
            txt_to_xlsx(man_txt, man_xlsx)
        data_man_ori = pd.read_excel(man_xlsx, sheet_name='Sheet1')
        data_man = data_man_ori.sort_values('speaker')
        data_man.to_excel(man_xlsx, index=False)
        data_man = pd.read_excel(man_xlsx, sheet_name='Sheet1')

        # correlation between automatic data and manual data for each dimension.
        if (data_auto['speaker'] == data_man['speaker']).all():
            print('Yes, speakers have same order in man and auto files.')
            rows = list(data_auto.columns)
            rows.remove('speaker')
            df_corr_auto_man = pd.DataFrame(np.arange((len(data_auto.columns) - 1) * 2).reshape(len(data_auto.columns) - 1, 2), index=rows, columns=['r', 'p'])
            for row in rows:
                x = data_auto[row].to_numpy()
                y = data_man[row].to_numpy()
                corr, p = scipy.stats.pearsonr(x, y)
                df_corr_auto_man.loc[row, 'r'] = round(corr, 3)
                df_corr_auto_man.loc[row, 'p'] = round(p, 5)
            if set_a == ['AA']:
                df_corr_auto_man.to_csv(filepath_exp + '/corr_man_auto_' + am + '_' + method + '_small.txt')
            else:
                df_corr_auto_man.to_csv(filepath_exp + '/corr_man_auto_' + am + '_' + method + '.txt')

            # plot scatter and regression line for data_auto and data_man.
            for i in range(int(len(feats)/4)):
                fig, ax = plt.subplots(4, 1, figsize=[5, 20])

                left = 0.125  # the left side of the subplots of the figure
                right = 0.9  # the right side of the subplots of the figure
                bottom = 0.05  # the bottom of the subplots of the figure
                top = 0.95  # the top of the subplots of the figure
                wspace = 0.05  # the amount of width reserved for blank space between subplots,
                hspace = 0.2  # the amount of height reserved for white space between subplots,
                plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                                    wspace=wspace, hspace=hspace)
                j = 0
                feat_group = feats[4*i]
                print('feat_group', feat_group)
                for feat in feats[4*i:4*i+4]:

                    x = data_man[feat]
                    y = data_auto[feat]
                    ax[j].scatter(x, y, c='b', s=80)

                    m, b = np.polyfit(x, y, 1)
                    corr, p = scipy.stats.pearsonr(x, y)
                    if p < 0.001:
                        p_str = '$^{***}$'
                    if 0.01 > p >= 0.001:
                        p_str = '$^{**}$'
                    if 0.05 > p >= 0.01:
                        p_str = '$^{*}$'
                    if p >= 0.05:
                        p_str = ''
                    ax[j].plot(x, m*x + b, 'r', label='r=' + str(round(corr, 2)) + p_str)
                    ax[j].legend(loc='upper left', fontsize=10)
                    feat_brief = feat.replace('_prc', '')
                    ax[j].set_title(feat_brief, fontsize=10)
                    ax[j].yaxis.get_major_formatter().set_powerlimits((0, 1)) # 将坐标轴的base number设置为一位。
                    ax[j].xaxis.get_major_formatter().set_powerlimits((0, 1))

                    if method == 'None' and j == 0:
                        fig2, bx = plt.subplots(1, 1)
                        bx.scatter(x, y, c='b', label='r=' + str(round(corr, 2)) + p_str)
                        bx.plot(x, m*x + b, 'r', label='y = '+str(round(m, 2)) + '*x + '+str(round(b, 2)))
                        xmin = np.min(x)
                        xmax = np.max(x)
                        dots_num = 15
                        xstep = (xmax-xmin)/dots_num
                        xx = np.arange(xmin, xmax, xstep)
                        bx.plot(xx, xx, color='grey', linestyle='dashed', label='y = x')
                        bx.legend(loc='upper left')
                        # feat_brief = feat.replace('_prc', '')
                        # bx.set_title(feat_brief, fontsize=10)
                        bx.yaxis.get_major_formatter().set_powerlimits((0, 1)) # 将坐标轴的base number设置为一位。
                        bx.xaxis.get_major_formatter().set_powerlimits((0, 1))
                        if set_a == ['AA']:
                            fig2.savefig(filepath_exp + '/man_auto_' + am + '_' + feat + '_' + method + '_small.pdf')
                        else:
                            fig2.savefig(filepath_exp + '/man_auto_' + am + '_' + feat + '_' + method + '.pdf')
                        plt.close(fig2)
                    # ax[0, 0].set_ylabel('automatic', fontsize=13)
                    # ax[0, 0].set_title('VAI', fontsize=15)
                    # ax[0, 1].set_title('VSA', fontsize=15)
                    # if i == int(len(feats_VAI_VSA)/2 - 1):
                    #     ax[i, j].set_xlabel('manual', fontsize=15)
                    #     ax[i, 0].set_ylabel('automatic', fontsize=15)
                    j = j + 1
                # plt.show()
                if set_a == ['AA']:
                    fig.savefig(filepath_exp + '/corr_man_auto_' + am + '_' + feat_group + '_' + method + '_small.pdf')
                else:
                    fig.savefig(filepath_exp + '/corr_man_auto_' + am + '_' + feat_group + '_' + method + '.pdf')
                plt.close(fig)

