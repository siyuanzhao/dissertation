# Basic libraries
from six.moves import cPickle as pickle #for performance
import pandas as pd
import numpy as np
import tensorflow as tf
from LSTMAutoencoder import LSTMAutoencoder
from random import shuffle
import sys
tf.set_random_seed(2016)
np.random.seed(2016)
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

# Constants
is_training = True
batch_num = 100
hidden_num = 50
iteration = 200
max_max_len = 2000
problem_set = 303899

model_name = str(problem_set)+'_jan31_gru_dropout'

if is_training:
    file_path = str(problem_set)+'_pl_train_data.csv'
    #file_path = 'sample_pl_train.csv'
else:
    file_path = str(problem_set) + '_train_data.csv'

# read csv file

pl_df = pd.read_csv(file_path)
# the number of features
elem_num = len(pl_df.columns)-2
# group by students
pl_df.set_index('id', inplace=True)
pl_g = pl_df.groupby('user_id')
cnt_list = []
for name,group in pl_g:
    cnt = len(group)
    cnt_list.append(cnt)
max_len = max(cnt_list)
avg_len = sum(cnt_list)/len(cnt_list)
print 'max len {}'.format(max_len)
print 'avg len {}'.format(avg_len)

max_len = min(max_len, max_max_len)
#max_len = 1000
for i in range(len(cnt_list)):
    if cnt_list[i] > max_len:
        cnt_list[i] = max_len

# get user id list
user_list = pl_df['user_id'].unique().tolist()

# placeholder list
p_input = tf.placeholder(tf.float32, [None, max_len, elem_num])

cell = tf.nn.rnn_cell.GRUCell(hidden_num)
#cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
if is_training:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)

ae = LSTMAutoencoder(hidden_num, p_input, max_len, cell=cell)

# should do the padding only once

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    if is_training:
        sess.run(tf.global_variables_initializer())
    else:
        # get representation for students
        saver.restore(sess, './'+model_name)
        chunks = [user_list[x:x+batch_num] for x in xrange(0, len(user_list), batch_num)]
        res = []
        for c in chunks:
            l = []
            batch_len = []
            gather_index = []
            for ite in c:
                m = pl_g.get_group(ite).iloc[:, :-1].as_matrix()
                if max_len >= m.shape[0]:
                    batch_len.append(m.shape[0])
                    diff = max_len - m.shape[0]
                    l.append(np.pad(m, ((0,diff), (0,0)), mode='constant', constant_values=0))
                else:
                    batch_len.append(max_len)
                    l.append(m[-1*max_len:, :])
                    #l.append(g.get_group(ite).iloc[:, 1:].as_matrix())
            x_batch = np.stack(l, axis=0)
            for i, ite in enumerate(batch_len):
                gather_index += range(i*max_len, i*max_len+ite-1)
            enc = sess.run(ae.enc_state,
                           {p_input:x_batch, ae.seq_len:batch_len,
                            ae.gather_index: gather_index})
            res.append(enc.h)
        res = np.concatenate(res)
        res_dict = dict(zip(user_list, res))
        save_dict(res_dict, str(problem_set)+'_result.pkl')
        print res_dict
        exit()

    min_loss = sys.maxint
    for i in range(iteration):
        # shuffle the user list
        shuffle(user_list)
        # divide user ids into batches
        chunks = [user_list[x:x+batch_num] for x in xrange(0, len(user_list), batch_num)]

        total_loss = 0
        for c in chunks:
            l = []
            batch_list = []
            gather_index = []
            for ite in c:
                m = pl_g.get_group(ite).iloc[:, :-1].as_matrix()
                if max_len >= m.shape[0]:
                    batch_list.append(m.shape[0])
                    diff = max_len - m.shape[0]
                    l.append(np.pad(m, ((0,diff), (0,0)), mode='constant', constant_values=0))
                else:
                    batch_list.append(max_len)
                    l.append(m[-1*max_len:, :])
                    #l.append(g.get_group(ite).iloc[:, 1:].as_matrix())
            x_batch = np.stack(l, axis=0)
            for idx, ite in enumerate(batch_list):
                gather_index += range(idx*max_len, idx*max_len+ite-1)

            loss_val, out, _, global_step = sess.run(
                [ae.loss, ae.output_, ae.train, ae.global_step],
                {p_input:x_batch, ae.seq_len:batch_list, ae.gather_index: gather_index})
            total_loss += loss_val
        print "iter %d:" % (i+1), total_loss
        if (i+1)% 10 == 0 and min_loss > total_loss:
            min_loss = total_loss
            saver.save(sess, './'+model_name)
