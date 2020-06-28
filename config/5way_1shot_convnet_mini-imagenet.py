from collections import OrderedDict

config = OrderedDict()

config['dataset_name'] = 'mini-imagenet'
config['num_generation'] = 5
config['num_loss_generation'] = 3
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'


config['emb_size'] = 128
config['backbone'] = 'convnet'

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['batch_size'] = 40
train_opt['iteration'] = 100000
train_opt['lr'] = 1e-3
train_opt['weight_decay'] = 1e-6
train_opt['dec_lr'] = 15000
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.1
train_opt['loss_indicator'] = [1, 1, 0]

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 1
eval_opt['batch_size'] = 10
eval_opt['iteration'] = 1000
eval_opt['interval'] = 1000

config['train_config'] = train_opt
config['eval_config'] = eval_opt
