import tensorflow as tf
import numpy as np

"""
Future : Modularization
"""

class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

  def __init__(self, hidden_num, p_input, max_len,
               cell=None, optimizer=None, reverse=False,
               decode_without_input=False, embedding_size=10, trainable_embed=False,
               ps_cnt=None):
    """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

    #self.batch_num = inputs[0].get_shape().as_list()[0]
    #inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, max_len, 1)]
    #inputs = tf.unstack(p_input, max_len, 1)
    inputs = p_input
    self.seq_len = tf.placeholder(tf.int32, [None])
    self.gather_index = tf.placeholder(tf.int32, [None])

    self.batch_num = tf.shape(inputs)[0]
    batch_size = inputs.get_shape().as_list()[0]
    feature_num = inputs.get_shape().as_list()[2]
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    if cell is None:
      self._enc_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
      self._dec_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
    else:
      self._enc_cell = cell
      self._dec_cell = cell

    if trainable_embed:
      self.elem_num = embedding_size*feature_num
      self.sq_embed_idx = tf.placeholder(tf.int32, [None, max_len])
      W = tf.get_variable('W', shape=[ps_cnt+1, embedding_size],
                          initializer=tf.contrib.layers.xavier_initializer())
      sq_embed = tf.nn.embedding_lookup(W, self.sq_embed_idx)
      cell_input = tf.reshape(tf.expand_dims(sq_embed, -2) * tf.expand_dims(p_input, -1),
                                 [-1, max_len, self.elem_num])
    else:
      self.elem_num = feature_num
      cell_input = p_input
    with tf.variable_scope('encoder'):
      self.z_codes, self.enc_state = tf.nn.dynamic_rnn(
        self._enc_cell, cell_input, self.seq_len, dtype=tf.float32)

    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="dec_bias")

      if decode_without_input:
        #dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        #              for _ in range(len(inputs))]
        dec_inputs = tf.zeros_like(inputs)
        dec_outputs, dec_state = tf.nn.dynamic_rnn(
          self._dec_cell, dec_inputs,
          initial_state=self.enc_state, dtype=tf.float32)
        """the shape of each tensor
          dec_output_ : (step_num x hidden_num)
          dec_weight_ : (hidden_num x elem_num)
          dec_bias_ : (elem_num)
          output_ : (step_num x elem_num)
          input_ : (step_num x elem_num)
        """
        if reverse:
          dec_outputs = tf.reverse(dec_outputs, [1])
        #dec_output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])
        dec_output_ = dec_outputs
        dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
        self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_

      else:
        dec_state = self.enc_state
        shape_list = inputs.get_shape().as_list()
        dec_input_ = tf.zeros([self.batch_num,self.elem_num], dtype=tf.float32)
        dec_outputs = []
        for step in range(shape_list[1]):
          if step>0:
            vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.sigmoid(tf.matmul(dec_input_, dec_weight_) + dec_bias_)
          #dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
          dec_outputs.append(dec_input_)
        if reverse:
          dec_outputs = dec_outputs[::-1]
        self.output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])

    # Indexing
    self.input_ = tf.gather(tf.reshape(
      cell_input, [-1, self.elem_num]), self.gather_index)
    self.output_ = tf.gather(tf.reshape(
      self.output_, [-1, self.elem_num]), self.gather_index)

    #self.input_ = inputs
    self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

    if optimizer is None:
      #optimizer = tf.train.AdamOptimizer(0.001)
      optimizer = tf.train.RMSPropOptimizer(0.001)
      grads_and_vars = optimizer.compute_gradients(self.loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
      grads_and_vars = [(tf.clip_by_norm(g, 10), v)
                          for g, v in grads_and_vars if g is not None]
      self.train = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=self.global_step)
    else:
      self.train = optimizer.minimize(self.loss, global_step=self.global_step)
