import pyabc
from pyabc import (IntegratedModel, ModelResult,
                   QuantileEpsilon)
import fitmulticell.model as morpheus_model
from scipy import stats
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import fitmulticell.sumstat.base as bs
from pyabc.sampler import RedisEvalParallelSampler
import argparse
from pyabc.sampler import MulticoreEvalParallelSampler

import fcntl, os, time


def setup_log_file_ER(path):
    counter_file = open(path, 'w')
    counter_file.write('0')  # Store 0 as starting number
    counter_file.close()


def update_early_rejected_counter(path):
    counter_file = open(path, 'r+')
    fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)
    count = int(counter_file.readline()) + 1
    counter_file.seek(0)
    counter_file.write(str(count))
    counter_file.close()


path_ER = "/home/emad/Documents/test_liver_model_delete/ER_paricles_log.txt"
setup_log_file_ER(path_ER)

parser = argparse.ArgumentParser(description='Parse necessary arguments')
parser.add_argument('-pt', '--port', type=str, default="50004",
                    help='Which port should be use?')
parser.add_argument('-ip', '--ip', type=str,
                    help='Dynamically passed - BW: Login Node 3')
args = parser.parse_args()

file_ = "/home/emad/Documents/test_liver_model_delete/YAP_Signaling_Liver_Regeneration_Model_reparametrized.xml"

par_map = {'k1': './CellTypes/CellType/Constant[@symbol="k1"]',
           'k2': './CellTypes/CellType/Constant[@symbol="k2"]',
           'k3_0': './CellTypes/CellType/Constant[@symbol="k3_0"]',
           'k4': './CellTypes/CellType/Constant[@symbol="k4"]',
           'k5': './CellTypes/CellType/Constant[@symbol="k5"]',
           'k6': './CellTypes/CellType/Constant[@symbol="k6"]',
           'k7': './CellTypes/CellType/Constant[@symbol="k7"]',
           'k8': './CellTypes/CellType/Constant[@symbol="k8"]',
           'k9': './CellTypes/CellType/Constant[@symbol="k9"]',
           'k10': './CellTypes/CellType/Constant[@symbol="k10"]',
           'k11': './CellTypes/CellType/Constant[@symbol="k11"]',
           'K_M1': './CellTypes/CellType/Constant[@symbol="K_M1"]',
           'K_M2': './CellTypes/CellType/Constant[@symbol="K_M2"]',
           'intensity_normalization_total': './CellTypes/CellType/Constant[@symbol="intensity_normalization_total"]',
           }

# tp_arg_list = [None, "Celltype", [1, 2, 3]]

model = morpheus_model.MorpheusModel(
    file_, par_map=par_map, par_scale="linear",timeout=10,
    show_stdout=False, show_stderr=False,
    raise_on_error=False,
    selected_time_points=[0,8,15,20,30,50],
    nr_data_points=10,
    ignore_list=["cell.id", "Tension"])

obs_pars = {'k1': 100,
            'k2': 2.02,
            'k3_0': 1.7,
            'k4': 0.19,
            'k5': 100,
            'k6': 0.18,
            'k7': 100,
            'k8': 100,
            'k9': 0.17,
            'k10': 1.8,
            'k11': 30,
            'K_M1': 0.0008,
            'K_M2': 0.25,
            'intensity_normalization_total': 1}

# generate data
model.par_scale = "log10"
obs_pars_log = {key: math.log10(val) for key, val in obs_pars.items()}
sim = model.sample(obs_pars)
observe_data_path = "/home/emad/Documents/test_liver_model_delete/YAP_Signaling_Liver_Regeneration_Data_edited_8.csv"
data = pd.read_csv(observe_data_path, sep=',')
dict_data = {}
for col in data.columns:
    dict_data[col] = data[col].to_numpy()

import csv
data2 = csv.DictReader(open(observe_data_path))


limits = {key: (math.log10((10 ** -1) * val), math.log10((10 ** 1) * val)) for
          key, val in obs_pars.items()}
limits["k1"] = (1, 3)
limits["k5"] = (1, 4)
limits["k7"] = (1, 4)
limits["k8"] = (1, 4)
limits["k2"] = (0,2)
limits["K_M1"] = (-4,-2)

prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub - lb)
                              for key, (lb, ub) in limits.items()})


redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port, look_ahead=False, wait_for_all_samples=True)

acceptor = pyabc.StochasticAcceptor(pdf_norm_method=ScaledPDFNorm())
kernel = pyabc.IndependentNormalKernel(
    var=[0.061763933333333]*60+ [0.050105066666667]*60, 
    keys=["IdSumstat__YAP_nuclear_observable","IdSumstat__YAP_total_observable"])
eps = pyabc.Temperature()

abc = pyabc.ABCSMC(model, prior, kernel, population_size=1000,
                   acceptor=acceptor, eps=eps, all_accepted=False)

db_path = "sqlite:///" + "/home/emad/Documents/test_liver_model_delete/" + "test_14param_Felipe.db"
history = abc.new(db_path, dict_data)
abc.run(max_nr_populations=40)

pyabc.visualization.plot_epsilons(history)
df, w = history.get_distribution(t=history.max_t)
plt.savefig(
    '/home/emad/Documents/test_liver_model_delete/Liver_regeneration_eps_14para_Felipe.png')

pyabc.visualization.plot_kde_matrix(df, w, limits=limits, refval=obs_pars_log)
plt.savefig(
    '/home/emad/Documents/test_liver_model_delete/Liver_regeneration_kde_mat_14para_Felipe.png')

