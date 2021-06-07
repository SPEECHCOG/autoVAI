#!/usr/bin/env python
# coding: utf-8

# It's a top-level program, which calls many methods defined in the same folder.
# to automatically compute the vowel articulation features of audio files:
# 1) put the audios (.wav) you want to measure in the /data/user folder.
# 2) generate frame-level formants for the audios using praat script frame_formants_measure
# 3) execute step from 1 to 3, then you will get the output of formant estimation for each corner vowel, vowel articulation features.
## 3-1) output files in /exp/user/ folder: $speaker_read_vowels_None_ipa.pdf (distribution of selected corner vowel frames in F2-F1 plane), speaker_formants_stat_auto_None_ipa.txt (formant estimates and vowel articulation features)

import os
from glob import glob
from pytictoc import TicToc
import time
from ipa_recognizer.recognizer import Recognizer
from soft_decode_ipa import soft_decode_ipa
from frames_screen import frames_screen
from vowel_articulation_auto import vowel_articulation_auto
from vowel_articulation_man import vowel_articulation_man
from auto_man_corr import auto_man_corr
from vowel_articulation_expert_rating_corr_PDSTU import vowel_articulation_expert_rating_corr_PDSTU
from vowel_articulation_expert_rating_corr_PCGITA import vowel_articulation_expert_rating_corr_PCGITA
from vowel_articulation_expert_rating_corr_TORGO import vowel_articulation_expert_rating_corr_TORGO
from formants_error_detection import formant_errors
from efficacy_frames_screen import manual_frames_cover_ratio
from efficacy_frames_screen import touched_segments

# hyperparameters definition.
tasks = ['PDSTU_read', 'PC-GITA_read', 'TORGO_52', 'user']
posterior_threshold = 0.2
# blank_factor, in range of [0.5, 2.0], the larger the more '<blk>' for ipa_recognizer output.
blank_factor = 0.5
set_a = ['AA', 'AE', 'AH', 'AW', 'AY']
set_i = ['IH', 'IX', 'IY']
set_u = ['UW', 'UH', 'OW']
# if only use corner vowels rather than an expanded phone category:
# set_a = ['AA']
# set_i = ['IY']
# set_u = ['UW']
am = 'ipa' # for automatic computation using Allosaurus.
# am = 'man' # for manual computation.
filepath_source = os.path.abspath(os.path.join(os.getcwd(), ".."))
# tested audios (.wav) should be put in folder of filepath_source/data/task
# experimental output files would be put in folder of filepath_source/exp/task
# codes are put in folder of filepath_source/program(_compact)

model = Recognizer()
phn_eng = model.decoder.unit.id_to_unit
t = TicToc()
print(filepath_source)

user_mode = input('Do you want user_mode (yes or no?):')
if user_mode == 'yes':
    task = 'user'
    wavs = glob(os.path.join(filepath_source, 'data', task, '*.wav'))
    no_wavs = 'no'
    while len(wavs) == 0 or no_wavs == 'no':
        no_wavs = input('Have you put tested audios (.wav) in /data/user/ folder? (yes, no)')
        wavs = glob(os.path.join(filepath_source, 'data', task, '*.wav'))

    frame_formants_exist = os.path.exists(os.path.join(filepath_source, 'exp', task, 'frame_formants_ipa.txt'))
    no_frame_formants = input('Have you computed the frame-level formants for tested audios using praat script "frame_formants_measure"? (yes or no)')
    if no_frame_formants == 'no' or frame_formants_exist == 'False':
        print('please check the value of filepath_source in frame_formants_measure.')
        time.sleep(2)
        check_filepath_source = input('Have you checked filepath_source in frame_formants_measure? (yes or no?)')
        # Issue: when call praat script from python, the code of form..endform which gets arguments from user cannot be executed.
        if check_filepath_source == 'yes':
            os.system('"/usr/bin/praat" --run frame_formants_measure')

    use_man = input('Do you have manual annotation for corner vowels saved in a .TextGrid file? (yes or no?)')
    if use_man == 'yes':

        man_formants_exist = os.path.exists(os.path.join(filepath_source, 'exp', task, 'vowel_formants_man.txt'))
        no_man_formants = input(
            'Have you computed mean formants for manually annotated segments in tested audios using praat script "formants_measure_man"? (yes or no)')
        if man_formants_exist == 'False' or no_man_formants == 'no':
            print('please check the value of filepath_source in formants_measure_man.')
            time.sleep(2)
            check_filepath_source = input('Have you checked the filepath_source in formants_measure_man? (yes or no?)')
            # Issue: when call praat script from python, the code of form..endform which gets arguments from user cannot be executed.
            if check_filepath_source == 'yes':
                os.system('"/usr/bin/praat" --run formants_measure_man')
        more_wavs = input('Do you have more than two tested audios? (yes, no)')
        if more_wavs == 'yes':
            num_steps = 5
        else:
            num_steps = 4
    elif use_man == 'no':
        num_steps = 3


for step in range(num_steps):
    step += 1

    if step == 1:
        print('step-1: soft decode the test audios using the phoneme recognizer Allosaurus...')
        t_start = t.tic()
        soft_decode_ipa(filepath_source, task, model, phn_eng, blank_factor)
        t_end = t.toc()
    if step == 2:
        print('step-2: select frames associated with corner vowels...')
        t_start = t.tic()
        frames_screen(filepath_source, task, am, posterior_threshold, set_a, set_i, set_u)
        t_end = t.toc()

    if step == 3:
        print('step-3: automatic computation for vowel articulation features...')
        t_start = t.tic()
        # before this step, frame_formants_measure has to be executed to generate frame-level formants.
        vowel_articulation_auto(filepath_source, task, am, set_a)
        t_end = t.toc()

    if step == 4:
        print('step-4: manual computation for vowel articulation features...')
        # before this step, formants_measure_man has to be executed to generate average formants for each manually annotated segment.
        t_start = t.tic()
        vowel_articulation_man(filepath_source, task)
        t_end = t.toc()
    if step == 5:
        print('step-5: correlation between manually and automatically computed vowel articulation features...')
        t_start = t.tic()
        auto_man_corr(filepath_source, task, am, set_a)
        t_end = t.toc()
    # step 6 is for 'PDSTU_read', 'PC-GITA_read' and 'TORGO_52'.
    if step == 6:
        print('step-6: correlation between vowel articulation features and subjective (expert) assessment...')
        if task == 'PDSTU_read':
            vowel_articulation_expert_rating_corr_PDSTU(filepath_source, set_a, task)
        if task == 'PC-GITA_read':
            vowel_articulation_expert_rating_corr_PCGITA(filepath_source, task)
        if task == 'TORGO_52':
            vowel_articulation_expert_rating_corr_TORGO(filepath_source, task)
    # step 7 and step 8 are for task = 'PDSTU_read'.
    if step == 7:
        print('step-7, formant errors detection for selected candidate frames...')
        t_start = t.tic()
        formant_errors(filepath_source, task, am)
        t_end = t.toc()
    if step == 8:
        print('step-8, efficacy of automatic frames selection for PDSTU_read (with manual annotations of corner vowels)...')
        t_start = t.tic()
        if task == 'PDSTU_read':
            manual_frames_cover_ratio(filepath_source, task, am)
            touched_segments(filepath_source, task, am)
        t_end = t.toc()


