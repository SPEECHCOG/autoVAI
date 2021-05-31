# This program is to soft decode read speech using ipa-recognizer, which is to assign each frame with multiple phonemes.
# LIU YUANYUAN, TUT, 2020-4-20.

import os
import numpy as np
from ipa_recognizer import Recognizer
from glob import glob

# We use default value of blank_factor to decode wav files, the larger the parameter, the more '<blk>' we get from the output.
def soft_decode_ipa(filepath_source, task, model, phn_eng, blank_factor):
    # model = Recognizer()
    # phn_eng = model.decoder.unit.id_to_unit
    # phn_eng = phn_dict.values()

    # filepath_source = os.path.abspath(os.path.join(os.getcwd(), ".."))
    filepath_task = filepath_source + '/data/' + task
    filepath_target = filepath_source + '/exp/' + task
    # print(filepath_target)
    frame_shift = 30
    f = open(os.path.join(filepath_target, 'ipa_soft_decode.txt'), 'w')
    f.writelines(['speaker\tframe\trec\tsum\tphn_1\tprob_1\tphn_2\tprob_2\tphn_3\tprob_3\tphn_4\tprob_4\t\n'])
    num = 1

    wav_files = glob(os.path.join(filepath_task, '*.wav'))
    # print('wav files:', wav_files)
    for wav_name in wav_files:
        print(wav_name)
        wav = os.path.basename(wav_name)
        if task == 'TORGO_52' or task == 'user':
            speaker = wav[0:wav.index('.')]
        else:
            speaker = wav[0:wav.index('_')]
        print(str(num), '\t', speaker)
        num = num + 1
        # print('start decoding with blank_factor=0.5...')
        # frame length 45ms and shift 30ms.
        out_phn = model.recognize(wav_name, blank_factor=blank_factor, no_remove=True)
        out_phn = out_phn.split(' ')
        # print('start generating logits with blank_factor=0.5...')
        out_score = np.array(model.logits(wav_name, blank_factor=blank_factor))[:, 0:40]
        # print('finish decoding.')
        # use softmax to generate probability from logits output.
        denorminator_one = np.sum(np.exp(out_score), axis=1).reshape(out_score.shape[0], 1)
        denorminator = np.repeat(denorminator_one, 40, axis=1)
        out_prob = np.exp(out_score) / denorminator
        prob_sum = np.sum(out_prob, axis=1)
        # out_prob = np.exp(out_score)/(1+np.exp(out_score))
        order_ascending = np.argsort(-out_prob, axis=1)
        order_top = order_ascending[:, 0:4]
        for i in range(order_top.shape[0]):
            f.writelines([speaker, '\t', str(i), '\t', out_phn[i], '\t', str(round(prob_sum[i], 3)), '\t'])
            for j in order_top[i]:
                phn = phn_eng[j]
                prob = np.round(out_prob[i, j], 3)
                f.writelines([phn, '\t', str(prob), '\t'])
            f.writelines('\n')
