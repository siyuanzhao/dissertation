
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats
import os
from numpy import std, mean, sqrt
import cPickle as pickle
import argparse
import sys
parser = argparse.ArgumentParser(description='Hyper parameters for the model')
parser.add_argument('-folder_path', action="store", default='', dest="folder_path", type=str)
parser.add_argument('-ps', action="store", default=255116, dest="ps", type=int)
opts = parser.parse_args(sys.argv[1:])

# In[2]:

def load_data(ps, ftype='test'):
    # load either test data or train data
    test_df = pd.read_csv('cfrnet/data/'+str(ps)+'_'+ftype+'_exp.csv', header=None)
    test_df.rename(columns={2: 'condition', 3: 'completion'}, inplace=True)
    print 'The avg completion rate in treatment {}'.format(test_df[test_df['condition'] == 1]['completion'].mean())
    print 'The avg completion rate in control {}'.format(test_df[test_df['condition'] == 0]['completion'].mean())
    return test_df


# ### Post-model analysis

# In[3]:

def post_analysis(res_df, test_df):
    #test_df = load_data(ps)
    #res_df = pd.read_csv(file_name, header=None)
    res_df = res_df.rename(columns={0: 'f', 1: 'cf'})
    concated_test_df = pd.concat([test_df, res_df], axis=1)
    concated_test_df['treatment_effect'] = np.where(concated_test_df['condition']==1,
                                                    concated_test_df['f']-concated_test_df['cf'],concated_test_df['cf']-concated_test_df['f'])
    concated_test_df['potential_treatment_outcome'] = np.where(concated_test_df['condition']==1,
                                                               concated_test_df['f'],concated_test_df['cf'])
    concated_test_df['potential_control_outcome'] = np.where(concated_test_df['condition']==0,
                                                             concated_test_df['f'], concated_test_df['cf'])
    # recommended condition
    concated_test_df['recommended_condition'] = np.where(concated_test_df['treatment_effect']>0, 1, 0)
    return concated_test_df


# In[4]:

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    print 'x group: # {} \t mean {} \t std {}'.format(nx, mean(x), std(x, ddof=1))
    print 'y group: # {} \t mean {} \t std {}'.format(ny, mean(y), std(y, ddof=1))
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


# In[22]:

def calculate_completion(res_df, data_df, verbose=1, i_subset=None, path=None):
    concated_df = post_analysis(res_df, data_df)
    if i_subset is not None:
        concated_df = concated_df.iloc[concated_df.index.get_indexer(i_subset)]
    matched_df = concated_df[((concated_df['condition'] == 1) & (concated_df['recommended_condition'] == 1))\
                             | ((concated_df['condition'] == 0) & (concated_df['recommended_condition'] == 0))]
    unmatched_df = concated_df[((concated_df['condition'] == 1) & (concated_df['recommended_condition'] == 0))\
                               | ((concated_df['condition'] == 0) & (concated_df['recommended_condition'] == 1))]

    def print_out(x, y, x_name, y_name, path):
        print stats.ttest_ind(x,y)
        print x_name
        print len(x)
        print y_name
        print len(y)
        print 'Effect size: '
        print cohen_d(x.tolist(), y.tolist())
        print '*'*10
        with open(path, 'a') as f:
            f.write('Comparison between {} and {}\n'.format(x_name, y_name))
            f.write('p-value {}\n'.format(stats.ttest_ind(x,y)[1]))
            f.write('{}: # {} \t mean {} \t std {}\n'.format(x_name, len(x), np.mean(x), np.std(x, ddof=1)))
            f.write('{}: # {} \t mean {} \t std {}\n'.format(y_name, len(y), np.mean(y), np.std(y, ddof=1)))
            f.write('Effect size: {}\n'.format(cohen_d(x.tolist(), y.tolist())))
            f.write('*'*10)
            f.write('\n')

    if verbose:
        print 'Comparison between treatment and control'
        print_out(concated_df[concated_df['condition']==1]['completion'],
                  concated_df[concated_df['condition']==0]['completion'], 'Treatment group', 'Control group', path)
        
        print 'Comparison between matched and unmatched'
        print_out(matched_df['completion'], unmatched_df['completion'], 'Matched group', 'Unmatched group', path)

        print 'Comparison between matched and actual treatment'
        print_out(matched_df['completion'],
                  concated_df[concated_df['condition']==1]['completion'], 'Matched group', 'Actual treatment group', path)

        print 'Comparison between matched and actual control'
        print_out(matched_df['completion'],
                  concated_df[concated_df['condition']==0]['completion'], 'Matched group', 'Actual control group', path)
    
    cr = matched_df['completion'].mean()
    return concated_df[['condition', 'recommended_condition',
                        'completion', 'potential_treatment_outcome', 'potential_control_outcome', 'treatment_effect']], cr


# In[23]:

def generate_final_table(test_ps, test_folder_path, folder_name, i_out):
    file_name = test_folder_path + folder_name + '/result.test.npz'
    # load test data
    data_df = load_data(test_ps)
    # load predictions on test
    result = load_result_file(file_name)
    preds = pd.DataFrame(result['pred'][:,:,0,i_out])
    post_df, _ = calculate_completion(preds, data_df,
                                      path=test_folder_path+'/'+str(test_ps)+'-p-values')
    post_df.to_csv(test_folder_path + '/'+str(test_ps)+'-final-table.csv', index=False)


# In[7]:

def load_result_file(file):
    arr = np.load(file)
    D = dict([(k, arr[k]) for k in arr.keys()])
    return D


# In[8]:

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# In[9]:

def find_best_config(folder_path):
    sorted_config_file = folder_path + '/configs_sorted.txt'
    config_dict = {}
    with open(sorted_config_file) as f:
        for line in f:
            for ite in line.split(','):
                ite = ite.strip()
                pair = ite.split('=')
                config_dict[pair[0]] = float(pair[1])
            break
    return config_dict


# In[10]:

def load_evaluation(eval_file):
    eval_results, configs = pickle.load(open(eval_file, "rb"))
    i_sel = np.argmin(eval_results['valid']['policy_risk'], 2)
    
# In[33]:

ps = opts.ps
folder_path = opts.folder_path
config_dict = find_best_config(folder_path)
data_df = load_data(ps, 'train')
cr_max = 0
result_name = ''
max_i_out = 0
for root, dirs, files in os.walk(folder_path):
    for name in dirs:
        if 'results_2' in name:
            config_file = folder_path + name + '/config.txt'
            res_config = {}
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    pair = line.split(':')
                    if is_number(pair[1]):
                        res_config[pair[0]] = float(pair[1])
            found = True
            # check if matched the best config
            for key in config_dict.keys():
                if res_config[key] != config_dict[key]:
                    found = False
                    break
            if found:
                result_name = name
                result_file = folder_path + name + '/result.npz'
                result = load_result_file(result_file)
                preds = result['pred']
                n_units, _, n_rep, n_outputs = preds.shape
                i_subset = result['val'][0].tolist()
                for i_out in range(n_outputs):
                    try:
                        _, cr = calculate_completion(pd.DataFrame(preds[:,:,0,i_out]), data_df, 0, i_subset)
                        if cr > cr_max:
                            cr_max = cr
                            max_i_out = i_out
                    except Exception as e:
                        print(e)
                        break

# once the best config is found then compute the results on testing data
generate_final_table(ps, folder_path, result_name, max_i_out)
