import tensorflow as tf
from cfr.util import load_assistments_data
import numpy as np
from sklearn.metrics import roc_auc_score

hidden_size = 500
output_size = 1
l2_lambda = 0
ps = 303899
#embeddings = False
embeddings = True
learning_rate = 0.001
epochs = 200
batch_size = 30
eval_interval = 20

rname='../lstm-autoencoder/'+str(ps)+'_result.pkl'
train_data_path = 'data/'+str(ps)+'_train_exp.csv'
test_data_path = 'data/'+str(ps)+'_test_exp.csv'
train_data = load_assistments_data(train_data_path, rname, embeddings)
test_data = load_assistments_data(test_data_path, rname, embeddings)

n_train = train_data['n']
n_test = test_data['n']
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]
input_size = train_data['dim']

with tf.name_scope("train"):
    # placeholders
    x = tf.placeholder(tf.float32, [None, input_size], name='input')
    output = tf.placeholder(tf.float32, [None, output_size], name='output')

    w1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.35),
                     name="weights1")
    bias1 = tf.Variable(tf.random_normal([hidden_size]))

    h = tf.nn.relu(tf.add(tf.matmul(x, w1),bias1))

    w2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.35),
                     name="weights2")
    bias2 = tf.Variable(tf.random_normal([output_size]))

    logits = tf.add(tf.matmul(h, w2),bias2)

    prob = tf.sigmoid(logits)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=logits)
    cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

    # loss op
    varss = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in varss])
    loss_op = cross_entropy_sum + l2_lambda*lossL2

def train_step(batch_x, batch_y):
    feed_dict = {x: batch_x, output: batch_y}
    _, step, cost = sess.run([train_op, global_step, cross_entropy_sum], feed_dict)
    return cost

def test_step(batch_x):
    feed_dict = {x: batch_x}
    step, preds = sess.run([global_step, prob], feed_dict)
    return preds

with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # decay learning rate
    starter_learning_rate = learning_rate
    #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_op)

    grads_and_vars = [(tf.clip_by_norm(g, 10), v)
                      for g, v in grads_and_vars if g is not None]
    train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
    #train_op = optimizer.minimize(cross_entropy_sum)
    sess.run(tf.global_variables_initializer())
    best_auc = 0
    for i in range(1, epochs+1):
        np.random.shuffle(batches)
        total_cost = 0
        for start, end in batches:
            batch_x = train_data['x'][start:end]
            batch_y = train_data['yf'][start:end]
            batch_cost = train_step(batch_x, batch_y)
            total_cost += batch_cost
        print 'epoch {}: cost: {}\n'.format(i, total_cost)

        if i % eval_interval == 0:
            test_preds = []
            for start in range(0, n_test, batch_size):
                end = min(n_test, start+batch_size)
                preds = test_step(test_data['x'][start:end])
                for ite in preds:
                    test_preds.append(ite[0])

            # calculate auc
            auc = roc_auc_score(test_data['yf'], test_preds)
            best_auc = max(best_auc, auc)
            print 'epoch {}: auc on test: {}\n'.format(i, auc)
            print 'best auc so far: {}\n'.format(best_auc)
