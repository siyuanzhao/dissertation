import os
import numpy as np
from cfr.util import load_dict
from logger import Logger as Log

def load_result_file(file):
    arr = np.load(file)

    D = dict([(k, arr[k]) for k in arr.keys()])

    return D

def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def load_single_result(result_dir):
    if Log.VERBOSE:
        print 'Loading %s...' % result_dir

    config_path = '%s/config.txt' % result_dir
    has_config = os.path.isfile(config_path)
    if not has_config:
        print 'WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir)
        config = None
    else:
        config = load_config(config_path)

    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path)
    except:
        'WARNING: Couldnt load result file. Skipping'
        return None

    n_rep = np.max([config['repetitions'], config['experiments']])

    if len(train_results['pred'].shape) < 4 or train_results['pred'].shape[2] < n_rep:
        print 'WARNING: Experiment %s appears not to have finished. Skipping.' % result_dir
        return None

    if has_test:
        test_results = load_result_file(test_path)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}

def load_results(output_dir):

    if Log.VERBOSE:
        print 'Loading results from %s...' % output_dir

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        #@TODO: Implement
        pass

    # Multiple results
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]

    if Log.VERBOSE:
        print 'Found %d experiment configurations.' % len(exp_dirs)

    # Load each result folder
    results = []
    for dir in exp_dirs:
        dir_result = load_single_result(dir)
        if dir_result is not None:
            results.append(dir_result)

    return results

def load_assistments_data(datapath, rname='../LSTM-autoencoder/259379_result.pkl', embeddings=False):
    data = np.loadtxt(open(datapath,"rb"),delimiter=",")
    #x = data[:,2:-1]
    #t = data[:,1:2]
    #yf = data[:,-1:]
    if embeddings:
        xdict = load_dict(rname)
        user_list= list(data[:,1])
        x_list = []
        for ite in user_list:
            x_list.append(xdict[int(ite)])
        x = np.stack(x_list, axis=0)
        x = np.expand_dims(x, axis=-1)
    else:
        x = data[:, 5:-1]
        x = np.expand_dims(data[:, 5:-1], axis=-1)
    t = data[:, 2:3]
    yf = data[:, 3:4]
    ycf = None
    HAVE_TRUTH = False
    SPARSE = False
    es = None
    ate = None
    ymul = 1
    yadd = 0
    ycfs = None
    mu0s = None
    mu1s = None

    data = {'x':x, 't':t, 'e':es, 'yf':yf, 'ycf':ycf,
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul,
            'YADD': yadd, 'ATE': [], 'HAVE_TRUTH': HAVE_TRUTH,
            'SPARSE': SPARSE}
    return data
    
def load_data(datapath):
    """ Load dataset """
    arr = np.load(datapath)
    xs = arr['x']

    HAVE_TRUTH = False
    SPARSE = False

    if len(xs.shape)==1:
        SPARSE = True

    ts = arr['t']
    yfs = arr['yf']
    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    try:
        ycfs = arr['ycf']
        mu0s = arr['mu0']
        mu1s = arr['mu1']
        HAVE_TRUTH = True
    except:
        print 'Couldn\'t find ground truth. Proceeding...'
        ycfs = None; mu0s = None; mu1s = None

    data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd, 'ATE': ate.tolist(), 'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}

    return data