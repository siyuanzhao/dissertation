# Basic libraries
from six.moves import cPickle as pickle #for performance
import pandas as pd
import numpy as np
import tensorflow as tf
from LSTMAutoencoder import LSTMAutoencoder
from random import shuffle
import sys
import os
import argparse

tf.set_random_seed(2016)
np.random.seed(2016)
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
# get hyperparameters from terminal
parser = argparse.ArgumentParser(description='Hyper parameters for the model')
parser.add_argument('-is_training', action="store", default=1, dest="is_training", type=int)
parser.add_argument('-ps', action="store", default=255116, dest="problem_set", type=int)
opts = parser.parse_args(sys.argv[1:])

# Constants
trainable_embed = False
is_training = opts.is_training
batch_num = 100
hidden_num = 30
iteration = 150
is_ps = True
#max_max_len = 1500
problem_set = opts.problem_set
directory = 'lstm-autoencoder/'
model_folder = directory+'saved_models'
result_folder = directory+'results'
# create folder
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

if is_ps:
    model_name = model_folder+'/'+str(problem_set)+'_gru_dropout_reverse'
    ps_idx_file = directory+str(problem_set) + '_ps_index'
else:
    model_name = model_folder+'/this_one_gru_dropout_reverse_march6'
    ps_idx_file = directory+'this_one_ps_index'

#file_path = str(problem_set)+'_extra_pl_train_data.csv'
if is_training:
    #file_path = 'this_one_pl_train_data.csv'
    if is_ps:
        file_path = directory+str(problem_set)+'_sq_train_data.csv'
    else:
        file_path = directory+'this_one_train_data.csv'
else:
    if is_ps:
        file_path = directory+str(problem_set) + '_sq_train_data.csv'
    else:
        file_path = directory+str(problem_set) + '_sq_train_data.csv'

# read csv file
pl_df = pd.read_csv(file_path)
d_num = 3

# the number of features
elem_num = len(pl_df.columns)-d_num
# group by students
pl_df.set_index('id', inplace=True)
pl_g = pl_df.groupby('user_id')
cnt_list = []
for name,group in pl_g:
    cnt = len(group)
    cnt_list.append(cnt)
max_len = max(cnt_list)
avg_len = sum(cnt_list)/len(cnt_list)
max_max_len = int(np.percentile(cnt_list, 70))
#max_max_len = 280
print 'max len {}'.format(max_len)
print 'avg len {}'.format(avg_len)
print 'max max len {}'.format(max_max_len)

max_len = min(max_len, max_max_len)
#max_len = 1000
for i in range(len(cnt_list)):
    if cnt_list[i] > max_len:
        cnt_list[i] = max_len

# get user id list
user_list = pl_df['user_id'].unique().tolist()

# placeholder list
p_input = tf.placeholder(tf.float32, [None, max_len, elem_num])

enc_cell = tf.nn.rnn_cell.GRUCell(hidden_num)
#dec_cell = tf.nn.rnn_cell.GRUCell(hidden_num)
dec_cell = enc_cell
#cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
#if is_training:
#    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)

ps_list = []
with open(ps_idx_file) as f:
    for line in f:
        ps_list.append(int(line))

if trainable_embed:
    ae = LSTMAutoencoder(
        hidden_num, p_input, max_len, cell=[enc_cell, dec_cell], reverse=True, embedding_size=10,
        trainable_embed=True,ps_cnt=len(ps_list))
else:
    ae = LSTMAutoencoder(hidden_num, p_input, max_len, cell=[enc_cell, dec_cell], reverse=True, ps_cnt=len(ps_list))
# should do the padding only once
len_dict = {}
x_dict = {}
ps_dict = {}

for ite in user_list:
    m = pl_g.get_group(ite).iloc[:, :-1*(d_num-1)].as_matrix()
    seq_ids = pl_g.get_group(ite)['sequence_id'].tolist()
    embed_ids = []

    for seq_id in seq_ids:
        if seq_id in ps_list:
            tmp_idx = ps_list.index(seq_id)
            embed_ids.append(tmp_idx)
        else:
            embed_ids.append(len(ps_list))

    if max_len >= m.shape[0]:
        len_dict[ite] = m.shape[0]
        diff = max_len - m.shape[0]
        x_dict[ite] = np.pad(m, ((0,diff), (0,0)), mode='constant', constant_values=0)
        ps_dict[ite] = embed_ids + [0]*diff
        
    else:
        len_dict[ite] = max_len
        x_dict[ite] = m[-1*max_len:, :]
        ps_dict[ite] = embed_ids[-1*max_len:]
        
def prepare_data(c, len_dict, x_dict, ps_dict):
    l = []
    batch_len = []
    gather_index = []
    batch_embed = []
    for ite in c:
        batch_len.append(len_dict[ite])
        l.append(x_dict[ite])
        batch_embed.append(ps_dict[ite])

    x_batch = np.stack(l, axis=0)
    for i, ite in enumerate(batch_len):
        gather_index += range(i*max_len, i*max_len+ite-1)
    return batch_embed, x_batch, batch_len, gather_index
            
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
            batch_embed, x_batch, batch_len, gather_index = prepare_data(
                c, len_dict, x_dict, ps_dict)
            if trainable_embed:
                enc = sess.run(ae.enc_state,
                               {ae.sq_embed_idx:batch_embed,p_input:x_batch,
                                ae.seq_len:batch_len,
                                ae.gather_index: gather_index})
            else:
                enc = sess.run(ae.enc_state,
                               {p_input:x_batch, ae.seq_len:batch_len,
                                ae.gather_index: gather_index, ae.sq_embed_idx:batch_embed})
            
            res.append(enc)
        res = np.concatenate(res)
        res_dict = dict(zip(user_list, res))
        save_dict(res_dict, result_folder+'/'+str(problem_set)+'_result.pkl')
        exit()

    min_loss = sys.maxint
    for i in range(iteration):
        # shuffle the user list
        shuffle(user_list)
        # divide user ids into batches
        chunks = [user_list[x:x+batch_num] for x in xrange(0, len(user_list), batch_num)]

        total_loss = 0
        for c in chunks:
            batch_embed, x_batch, batch_list, gather_index = prepare_data(
                c, len_dict, x_dict, ps_dict)
                
            if trainable_embed:
                loss_val, out, _, global_step = sess.run(
                    [ae.loss, ae.output_, ae.train, ae.global_step],
                    {ae.sq_embed_idx:batch_embed, p_input:x_batch, ae.seq_len:batch_list,
                     ae.gather_index: gather_index})
            else:
                loss_val, out, _, global_step, input_, output_ = sess.run(
                    [ae.loss, ae.output_, ae.train, ae.global_step, ae.input_, ae.output_],
                    {p_input:x_batch, ae.seq_len:batch_list,
                     ae.gather_index: gather_index, ae.sq_embed_idx:batch_embed})
            total_loss += loss_val
        print "iter %d:" % (i+1), total_loss
        if (i+1)% 10 == 0 and min_loss > total_loss:
            min_loss = total_loss
            saver.save(sess, './'+model_name)
