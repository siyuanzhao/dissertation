import os
import pandas as pd
import numpy as np
import datetime
import random
import glob
# prepare experimental data
# input file: this_one.csv
# example of output file: ps_exp.csv, ps_train_exp.csv, ps_test_exp.csv

# hyperparameters
# using sea representation
is_sea = 1
all_ps_list = [241622,243393, 246482, 246647, 255116, 263052, 259379, 237447, 246627, 250476, 226210, 303899, 263115, 627695, 694197]
this_one_list = [241622,243393, 246482, 246647, 255116, 263052, 259379, 237447, 246627, 250476, 226210, 303899, 263115]
is_video_list = [False, False, False, False, False, True, True, False, False, True, False, False, True, True, False]
ps_video_dict = {}
for idx, ps in enumerate(all_ps_list):
    ps_video_dict[ps] = is_video_list[idx]
ps_list = [694197]

for idx, ps in enumerate(ps_list):
    is_video = ps_video_dict[ps]
    directory = '22-Experiments/' + str(ps)
    if not os.path.exists(directory):
        os.makedirs(directory)
    exp_file = '22-Experiments/' + str(ps) + '/' + str(ps) + '_exp.csv'
    exp_stat_file = '22-Experiments/' + str(ps) + '/' + 'stats'
    if not os.path.isfile(exp_file):
        csv_path = directory + '/' + str(ps) + '_exp.csv'
        df = pd.read_csv('22-Experiments/ThisOne.csv')
        cfr_df = pd.DataFrame()
        problem_set_id = ps
        video_text_df = df[df['problem_set'] == problem_set_id]

        print len(video_text_df)

        if is_video:
            video_text_df = video_text_df[((video_text_df['Saw Video'] == 1) & \
                                           (video_text_df['ExperiencedCondition'] == True) & (video_text_df['condition']=='E'))\
                                          | ((video_text_df['Saw Video'] == 1) & (video_text_df['ExperiencedCondition'] == True)\
                                             & (video_text_df['condition']=='C'))]
            cfr_df['user_id'] = video_text_df['User ID']
            cfr_df['condition'] = ((video_text_df['Saw Video'] == 1) & \
                                   (video_text_df['ExperiencedCondition'] == True) & (video_text_df['condition']=='E')).astype(int)
        else:
            video_text_df = video_text_df[((video_text_df['ExperiencedCondition'] == True) & (video_text_df['condition']=='E'))\
                                          | ((video_text_df['ExperiencedCondition'] == True) & (video_text_df['condition']=='C'))]
            #print video_text_df.head()
            cfr_df['user_id'] = video_text_df['User ID']
            cfr_df['condition'] = np.where((video_text_df['condition']=='E')&\
                                           (video_text_df['ExperiencedCondition']==True), 1, 0)

        cfr_df['yf'] = video_text_df['complete']

        cfr_df['ycf'] = np.NaN

        cols = ['Prior Problem Count', 'Prior Correct Count', 'Prior Assignments Assigned', \
                'Prior Assignment Count', 'Prior Homework Assigned', 'Prior Homework Count', \
                'Prior Completion Count', 'Prior Homework Completion Count', 'problem_count']
        cols_norm = map(lambda x: x+' zscore', cols)
        for idx, col in enumerate(cols):
            video_text_df[cols_norm[idx]] = (video_text_df[col] - video_text_df[col].min())/(video_text_df[col].max() - video_text_df[col].min())

        features_list = ['Prior Percent Correct', 'Prior Percent Completion', 'Z-Scored Mastery Speed',
                         'Prior Homework Percent Completion',
                         'Z-Scored HW Mastery Speed', 'Prior Class Homework Percent Completion',
                         'Prior Class Percent Completion'] + cols_norm

        for ite in features_list:
            cfr_df[ite] = video_text_df[ite]

        print cfr_df['condition'].value_counts()
        np.random.seed(2018)
        msk = np.random.rand(len(cfr_df)) < 0.5

        cfr_df.to_csv(csv_path, header=False, na_rep='0')
        print 'avg completion rate {}'.format(cfr_df['yf'].mean())
        train_df = cfr_df[msk]
        test_df = cfr_df[~msk]
        print len(train_df)
        print len(test_df)
        train_df.to_csv(directory + '/' + str(problem_set_id) +'_train_exp.csv', header=False, na_rep='0')
        test_df.to_csv(directory + '/' + str(problem_set_id) +'_test_exp.csv', header=False, na_rep='0')
        with open(exp_stat_file, 'w') as f:
            f.write('Train Data\n')
            f.write('Policy risk for control {}\n'.format(1-train_df[train_df['condition']==0]['yf'].mean()))
            f.write('Policy risk for treatment {}\n'.format(1-train_df[train_df['condition']==1]['yf'].mean()))
            f.write('Test Data\n')
            f.write('Policy risk for control {}\n'.format(1-test_df[test_df['condition']==0]['yf'].mean()))
            f.write('Policy risk for treatment {}\n'.format(1-test_df[test_df['condition']==1]['yf'].mean()))
        # copy files over to cfrnet
        os.system('cp 22-Experiments/'+str(ps)+'/'+str(ps)+'_*.csv cfrnet/data/')

    # prepare data from problem logs for SEA
    problem_set = ps
    directory = 'lstm-autoencoder/'
    student_cnt = 0
    read_csv = False
    limit_cnt = 300
    data_file = directory + str(problem_set)+'_sq_train_data.csv'
    ps_index = directory + str(problem_set)+'_ps_index'

    if not os.path.isfile(data_file) and is_sea:
        if ps in this_one_list:
            pl_df = pd.read_csv(directory+'this_one_problem_logs_seq.csv')
        else:
            pl_df = pd.read_csv(directory+str(ps)+'_problem_logs.csv')

        #pl_df['formatted_start_time'] = pd.to_datetime(pl_df['start_time'])

        #date_before = datetime.date(2016, 8, 1)

        #print 'the number of students {}'.format(len(pl_df['user_id'].unique()))

        #pl_df = pl_df[pl_df['formatted_start_time'] < date_before].reset_index()

        print 'the number of students {}'.format(len(pl_df['user_id'].unique()))
        # read one experiment
        ps_df = pd.read_csv('22-Experiments/'+str(problem_set)+'/'+str(problem_set)+'_exp.csv', header=None)

        sublist = random.sample(pl_df['user_id'].unique(), student_cnt)
        student_list = sublist + ps_df[1].unique().tolist()

        train_df = pl_df[pl_df['user_id'].isin(student_list)].reset_index()

        train_df = train_df[train_df['original'] == 1]

        print 'number of row in train_df {}'.format(len(train_df))
        print 'number of row in pl_df {}'.format(len(pl_df))
        if not os.path.isfile(ps_index):
            with open(ps_index, 'w') as f:
                sub_counts = pd.value_counts(train_df['sequence_id'])
                for ite in sub_counts[sub_counts > limit_cnt].index.tolist():
                    f.write(str(ite) + '\n')

        def prepare(sub_df):
            new_df = pd.DataFrame()
            # u'correct , u'bottom_hint, u'hint_count, u'attempt_count, u'first_response_time, first_action
            new_df['user_id'] = sub_df['user_id']
            # binary correctness
            new_df['correct'] = np.where(sub_df['correct'] < 1, 0, 1)
            new_df['bottom_hint'] = sub_df['bottom_hint']
            new_df['hint_count'] = sub_df['hint_count']
            new_df['attempt_count'] = sub_df['attempt_count']
            new_df['first_response_time'] = sub_df['first_response_time']
            # first action
            new_df['first_action'] = sub_df['first_action']

            #norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() - x.min() else x/(x+1)
            # normalize within problems
            #new_df.insert(len(new_df.columns), 'NofFRT', sub_df.groupby('sequence_id')['first_response_time'].transform(norm))
            new_df.insert(len(new_df.columns), 'NofFRT', sub_df.groupby('sequence_id')['first_response_time'].rank(pct=True))
            del new_df['first_response_time']

            new_df.insert(len(new_df.columns), 'normalized_hint_count', sub_df.groupby('sequence_id')['hint_count'].rank(pct=True))
            del new_df['hint_count']

            new_df.insert(len(new_df.columns), 'normalized_attempt_count', sub_df.groupby('sequence_id')['attempt_count'].rank(pct=True))
            del new_df['attempt_count']
            del new_df['user_id']
            new_df['sequence_id'] = sub_df['sequence_id']
            new_df['user_id'] = sub_df['user_id']
            new_df['id'] = sub_df['id']
            return new_df

        new_df = prepare(train_df)

        def checknull(df):
            for column in df.columns:
                if df[column].isnull().values.any():
                    print column

        checknull(new_df)
        new_df.to_csv(data_file, index=False)

        # train SEA
    directory = 'lstm-autoencoder/results/'
    result_file = directory+str(ps)+'_result.pkl'
    model_folder = 'lstm-autoencoder/saved_models'
    model_file = model_folder+'/'+str(ps)+'_gru_dropout_reverse*'
    if not glob.glob(model_file) and is_sea:
        os.system('python lstm-autoencoder/train.py -is_training 1 -ps '+str(ps))
    if not os.path.isfile(result_file) and is_sea:
        os.system('python lstm-autoencoder/train.py -is_training 0 -ps '+str(ps))
    os.system('./cfrnet/assistments_exp.sh {} {}'.format(ps, is_sea))
    folder_path = 'cfrnet/results/'+str(ps)+'/'
    os.system('python result_analysis.py -ps {} -folder_path {}'.format(ps, folder_path))
