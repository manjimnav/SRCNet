import argparse
import os
from src.experiment import Experimenter
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pandas as pd
import itertools
from src.models import get_model
import yaml

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dict_product(dicts):
    return (dict(zip([k for k, v in dicts.items() if v is not None], x)) for x in itertools.product(*[v for v in dicts.values() if v is not None]))

def seed_all():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_params(model, global_args):
    
    yaml_file = open(f"./modelconfs/{model}.yaml", 'r')
    model_args = yaml.safe_load(yaml_file)
    global_args.update(model_args)
    dict_experiment_params = dict_product(global_args)
    
    for args_e in dict_experiment_params:
        yield args_e


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, nargs='+', default=['CNN'])
    parser.add_argument('--results_file', type=str, default='metrics')
    parser.add_argument('--scale_results', action='store_true')

    args = parser.parse_args()
    

    yaml_file = open("./modelconfs/global.yaml", 'r')
    global_args = yaml.safe_load(yaml_file) 

    for model_name in args.model:
        
        for experiment_args in get_params(model_name, global_args):
            experiment_args['model'] = model_name
            if experiment_args['seq_len']<experiment_args['pred_len']:
                continue

            if os.path.exists(f'./results/{args.results_file}.csv'):
                metrics_df = pd.read_csv(f'./results/{args.results_file}.csv')
                
                if set(experiment_args.keys()).issubset(metrics_df.columns.tolist()) and (metrics_df[list(experiment_args.keys())] == experiment_args.values()).all(axis=1).any():
                    continue

            experiment_args['scale_results'] = args.scale_results

            experiment_args = dotdict(experiment_args)

            all_metrics = []
            print(experiment_args.items())

            training_time = 0
            experiment_hash = hash(frozenset(experiment_args.items()))

            seed_all()

            for it in range(experiment_args.iters):
                
                model = get_model(experiment_args)
                exp = Experimenter(experiment_args, model)
                print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
                model, tt = exp.train(f'{experiment_hash}/{it}')

                print('>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                metrics = exp.test(f'{experiment_hash}/{it}')
                all_metrics.append(metrics)
                training_time += tt/3

            all_metrics = np.mean(all_metrics, axis=0)

            dataset = experiment_args.data_path.replace(".csv", "")

            experiment_args.update({'training_time': training_time, 
                                    'mae': all_metrics[0], 'mse': all_metrics[1], 
                                    'rmse': all_metrics[2], 'mape': all_metrics[3],
                                    'mspe': all_metrics[4], 'wape': all_metrics[5],
                                    'model_params': count_parameters(get_model(experiment_args))})

            experiment_metrics = pd.DataFrame(experiment_args, index=[0])
            experiment_metrics['experiment_id'] = experiment_hash

            if os.path.exists(f'./results/{args.results_file}.csv'):
                metrics_df = pd.read_csv(f'./results/{args.results_file}.csv')
                experiment_metrics.index = [len(metrics_df)]
                experiment_metrics = pd.concat([metrics_df, experiment_metrics])

                if set(experiment_metrics.columns.tolist()).issubset(metrics_df.columns.tolist()):
                    experiment_metrics.iloc[-1].to_frame().T.to_csv(f'./results/{args.results_file}.csv', mode='a', header=None, index = None)
                else:
                    experiment_metrics.to_csv(f'./results/{args.results_file}.csv', index = None)
            else:
                experiment_metrics.to_csv(f'./results/{args.results_file}.csv', index = None)
        