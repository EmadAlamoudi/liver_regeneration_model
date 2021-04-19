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

path_ER = "/p/project/fitmulticell/emad/liver_regeneration_model_scripts/set_11/384_la_fixed/ER_paricles_log.txt"
setup_log_file_ER(path_ER)


parser = argparse.ArgumentParser(description='Parse necessary arguments')
parser.add_argument('-pt', '--port', type=str, default="50004",
                    help='Which port should be use?')
parser.add_argument('-ip', '--ip', type=str,
                    help='Dynamically passed - BW: Login Node 3')
args = parser.parse_args()


file_ = "/p/project/fitmulticell/emad/liver_regeneration_model_scripts/set_11/384_la_fixed/new_model/YAP_Signaling_Liver_Regeneration_Model_reparametrized.xml"

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
root_path = "/p/project/fitmulticell/emad/liver_regeneration_model_scripts/set_11/384_la_fixed"
model = morpheus_model.MorpheusModel(
    file_, par_map=par_map,par_scale="linear",
    show_stdout=False, show_stderr=False,timeout=900, 
    raise_on_error=False,executable="/p/project/fitmulticell/emad/morpheus-binary/morpheus-2.2.0-beta2",
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
# observed_data = model.sample(obs_pars)
model.par_scale = "log10"
obs_pars_log = {key: math.log10(val) for key, val in obs_pars.items()}

observe_data_path = root_path + "/YAP_Signaling_Liver_Regeneration_Data_edited_7.csv"
data = pd.read_csv(observe_data_path, sep=',')
dict_data = {}
for col in data.columns:
    dict_data[col] = data[col].to_numpy()

import csv

data2 = csv.DictReader(open(observe_data_path))
#observed_data = load_obj("obs_data_4_132t")


limits = {key: (math.log10((10**-1)*val), math.log10((10**1)*val)) for key, val in obs_pars.items()}
limits["k1"] = (1, 3)
limits["k5"] = (1, 4)
limits["k7"] = (1, 4)
limits["k8"] = (1, 4)
limits["k2"] = (0,1)
limits["K_M1"] = (-3.6,-2)

# limits_2 = dict()
# limits_2["k3_0"] = (-1, 1)
# limits_2["k4"] = (-1, 1)
# limits_2["k6"] = (-1, 1)
# limits_2["k9"] = (-1, 1)
# limits_2["k10"] = (-1, 1)
# limits_2["k11"] = (-1, 1)
# limits_2["intensity_normalization_total"] = (-1, 1)


prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub - lb)
                              for key, (lb, ub) in limits.items()})


def eucl_dist_Jan(sim, obs):
    total = 0
    for key in sim:
        if key == 'loc': continue
        x = np.array(sim[key])
        y = np.array(obs[key])
        if np.max(y) != 0:
            x = x/np.max(y)
            y = y/np.max(y)
        total += np.sum((x - y) ** 2) / x.size
    return total


def eucl_dist_Jan_2(sim, obs):
    if sim == -15:
        update_early_rejected_counter(path_ER)
        return np.inf
    sim_edit = prepare_data_3(obs, sim)
    total = 0
    for key in sim_edit:
        if key in ('loc', "IdSumstat__time", "IdSumstat__cell.id", "IdSumstat__Tension"):
            continue

        x = np.array(sim_edit[key])
        y = np.array(obs[key])
        z = np.array(obs["SEM_" + key])
        # simulation does not finish successfuly, only partial part of the
        # result wrtten. In such case, ignore the parameter vector
        if x.size != y.size:
            print("size not match")
            return np.inf
        total += np.sum(((x - y)/z) ** 2)
    return total

def prepare_data(obs, sim):
    unique_obs = np.unique(obs["time"])
    itemindex = []
    for val in unique_obs:
        itemindex = np.concatenate(
            [itemindex, np.where(sim["time"] == val)[0]])
    for key in sim:
        if key == "loc": continue
        sim[key] = sim[key][itemindex.astype(int)]
    
    return sim


def prepare_data_2(obs, sim):
    unique_obs = [0.0,0.8,1.5,2.0,3.0,5.0]
    itemindex = []
    new_dict = dict()
    uniqe_key_list = []
    for uniqe_key in [*sim]:
        uniqe_key_list.append(uniqe_key.split("__")[0])
    uniqe_key_list = list(OrderedDict.fromkeys(uniqe_key_list))
    uniqe_key_list.remove('loc')
    for key in sim:
        if key == "loc": continue
        for val in unique_obs:
            new_val = [value for key, value in sim.items() if str(val) in key.lower()]
            for i, new_val_index in enumerate(new_val,0):
                new_dict[uniqe_key_list[i]+"__"+str(val)] = new_val_index

    return new_dict

def prepare_data_3(obs, sim):
    unique_obs = [0,8,15,20,30,50]
    step = 10
    itemindex = []
    new_dict = dict()
    uniqe_key_list = []
    for key in [*sim]:
        if key == 'loc':
            continue
        new_dict[key]=[]
        for index in unique_obs:
            new_dict[key].extend(sim[key][index*step:(index*step)+step])
    return new_dict


# redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port, look_ahead=True,log_file="/home/ealamoodi/liver_regeneration_model_scripts_2/set_10/log/Liver_regeneration_log_14para_Felipe.csv")

redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port, look_ahead=True, log_file= root_path + "/log/Liver_regeneration_log_14para_Felipe.csv", max_n_eval_look_ahead_factor=2)



# early rejection model

class MyStochasticProcess(IntegratedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_early_stopped = 0

    def integrated_simulate(self, pars, eps):
        simulation = model.sample(pars=pars)
        if simulation == -15:
            return ModelResult(accepted=False)
        elif eps < eucl_dist_Jan_2(simulation, dict_data):
            return ModelResult(accepted=False)
        return ModelResult(accepted=True,
                           distance=eucl_dist_Jan_2(simulation, dict_data),
                           sum_stats=simulation)


model_early = MyStochasticProcess()


abc = pyabc.ABCSMC(model, prior, eucl_dist_Jan_2, population_size=1000,
                   eps=QuantileEpsilon(alpha=0.5),sampler=redis_sampler, all_accepted=False)


# abc = pyabc.ABCSMC(model, prior, eucl_dist_Jan_2, population_size=500,sampler=redis_sampler)
db_path = "sqlite:///" + root_path + "/db/" + "test_14param_Felipe.db"
history = abc.new(db_path, dict_data)
abc.run(min_acceptance_rate=0.001)


pyabc.visualization.plot_epsilons(history)
df, w = history.get_distribution(t=history.max_t)
plt.savefig(root_path + '/outplot/Liver_regeneration_eps_14para_Felipe.png')

pyabc.visualization.plot_kde_matrix(df, w, limits=limits, refval=obs_pars_log)
plt.savefig(root_path + '/outplot/Liver_regeneration_kde_mat_14para_Felipe.png')

# plt.show()


