from sklearn import linear_model
from cfrnet.cfr.util import load_assistments_data
import sys
import argparse
import numpy as np
import pandas as pd
import os
from result_analysis import calculate_completion_util

#reg = linear_model.LassoLars(alpha=.1)
reg = linear_model.LogisticRegression()

# load training data
parser = argparse.ArgumentParser(description='Hyper parameters for the model')
parser.add_argument('-ps', action="store", default=255116, dest="problem_set", type=int)
parser.add_argument('-ae', action="store", default=1, dest="ae", type=int)
opts = parser.parse_args(sys.argv[1:])
ps = opts.problem_set
embeddings = opts.ae

datapath = 'cfrnet/data/'+str(ps)+'_train_exp.csv'
datapath_test = 'cfrnet/data/'+str(ps)+'_test_exp.csv'
rname='lstm-autoencoder/results/'+str(ps)+'_result.pkl'
D = load_assistments_data(datapath, rname=rname, embeddings=embeddings)
D_test = None
D_test = load_assistments_data(datapath_test, rname=rname, embeddings=embeddings)

x = D['x']
t = D['t']
yf = np.squeeze(D['yf'])
x = np.concatenate((x,t), 1)
reg.fit(x, yf)
print(reg.coef_)
x_test = D_test['x']
t_test = D_test['t']
completion_test = D_test['yf']
x_test_treatment = np.concatenate((x_test, np.ones([x_test.shape[0], 1])), 1)
x_test_control = np.concatenate((x_test, np.zeros([x_test.shape[0], 1])), 1)
treatment_preds = np.expand_dims(reg.predict_proba(x_test_treatment)[:,1], axis=1)
control_preds = np.expand_dims(reg.predict_proba(x_test_control)[:,1], axis=1)
recommanded_condition = np.where((treatment_preds-control_preds) > 1, 1, 0)

final_res = np.concatenate((t_test, treatment_preds, control_preds, completion_test, recommanded_condition), 1)
# 0 - condition, 1 - treat, 2 - control, 3 - completion, 4 - recommanded_condition
concated_df = pd.DataFrame(final_res,
             columns=['condition', 'treatment_outcome', 'control_outcome',
                      'completion', 'recommended_condition'])
path = 'cfrnet/results/'+str(ps)
if not os.path.exists(path):
    os.makedirs(path)
if embeddings:
    p_path = path + '/'+str(ps)+'-AE-regression-p-values'
    t_path = path + '/'+str(ps)+'-AE-regression-final-table.csv'
else:
    p_path = path + '/'+str(ps)+'-numeric-regression-p-values'
    t_path = path + '/'+str(ps)+'-numeric-regression-final-table.csv'

calculate_completion_util(concated_df, 1, p_path)
concated_df.to_csv(t_path, index=False)
