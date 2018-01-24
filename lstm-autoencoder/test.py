# Basic libraries
import numpy as np
import tensorflow as tf
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import LSTMAutoencoder

# Constants
batch_num = 128
hidden_num = 12
step_num = 8
elem_num = 1
iteration = 1000

# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
#p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
ae = LSTMAutoencoder(hidden_num, p_input, step_num, cell=cell, decode_without_input=True, reverse=True)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.all_variables())
  min_loss = 200
  for i in range(iteration):
    """Random sequences.
      Every sequence has size batch_num * step_num * elem_num 
      Each step number increases 1 by 1.
      An initial number of each sequence is in the range from 0 to 19.
      (ex. [8. 9. 10. 11. 12. 13. 14. 15])
    """
    r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
    r = np.tile(r, (1, step_num, elem_num))
    d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
    d = np.tile(d, (batch_num, 1, 1))
    random_sequences = r+d
    loss_val, _, global_step = sess.run([ae.loss, ae.train, ae.global_step], {p_input:random_sequences})
    print "iter %d:" % (i+1), loss_val
    if min_loss > loss_val:
      min_loss = loss_val
      #saver.save(sess, 'checkpoints', global_step)

  input_, output_ =  sess.run([ae.input_, ae.output_], {p_input:r+d})
  print "train result :"
  print "input :", input_[0,:,:].flatten()
  print "output :", output_[0,:,:].flatten()
