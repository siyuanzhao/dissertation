import sys
import os

import cPickle as pickle

from cfr.logger import Logger as Log
Log.VERBOSE = True

import cfr.evaluation as evaluation
from cfr.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None, embeddings=False, rname=None, ps=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)

    output_dir = 'results/'+ps

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+'/'+ps + '_train_exp.csv'
    data_test = cfg['datadir']+'/'+ps + '_test_exp.csv'
    binary = False
    if cfg['loss'] == 'log':
        binary = True

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                                    data_path_train=data_train,
                                                    data_path_test=data_test,
                                                    binary=binary, embeddings=embeddings, rname=rname)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print 'Loading evaluation results from %s...' % eval_path
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Sort by alpha
    #eval_results, configs = sort_by_config(eval_results, configs, 'p_alpha')

    # Print evaluation results
    if binary:
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)

    # Plot evaluation
    #if configs[0]['loss'] == 'log':
    #    plot_cfr_evaluation_bin(eval_results, configs, output_dir)
    #else:
    #    plot_cfr_evaluation_cont(eval_results, configs, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>'
    else:
        config_file = sys.argv[1]
        ps = sys.argv[5]
        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        #if len(sys.argv)>3:
        #    filters = eval(sys.argv[3])
        embeddings = False
        if len(sys.argv)>3 and sys.argv[3] == '1':
            embeddings = True
        rname = None
        if len(sys.argv)>4:
            rname = sys.argv[4]

        evaluate(config_file, overwrite, filters=filters, embeddings=embeddings, rname=rname, ps=ps)
