from hpbandster.core.worker import Worker
import ConfigSpace as CS
import os
import pickle
import scipy
import time
import numpy
from hpbandster.optimizers import RandomSearch as RandomSearch
import hpbandster.core.result as hpres
import hpbandster.core.nameserver as hpns
from cnn_mnist import train_and_validate, mnist
import tensorflow as tf
import logging
logging.basicConfig(level=logging.WARNING)

# load data
x_train, y_train, x_valid, y_valid, x_test, y_test = mnist('./')

# basic implementation of a worker class


class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        # start tf session
        with tf.Session() as sess:
            # train model and keep results
            results, model = train_and_validate(x_train, y_train, x_valid, y_valid, budget, config['lr'], config['num_filters'], config['batch_size'], config['filter_size'])

        # report error = (1 - validation) from last epoch for random search to minimize, and report results
        return({
            'loss': 1 - results['val_acc'][-1],
            'info': {
                'config': config,
                'validation_accuracy': results['val_acc'][-1],
                'results': results
            }
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, default_value=1e-2, log=True))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=32, log=True))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=32, log=True))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('filter_size', [3, 5]))
        return(config_space)


# name server
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# worker
w = MyWorker(sleep_interval=0, nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# optimizer
randomSearch = RandomSearch(configspace=w.get_configspace(),
                            run_id='example1', nameserver='127.0.0.1',
                            min_budget=1, max_budget=6
                            )
res = randomSearch.run(n_iterations=50)

# store results for analysis
with open(os.path.join('.', 'results_random_search.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

# in the end, shutdown
randomSearch.shutdown(shutdown_workers=True)
NS.shutdown()

# print results
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.' % (sum([r.budget for r in res.get_all_runs()])/6))
