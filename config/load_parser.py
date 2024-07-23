# load_parser.py


import yaml
import argparse
import collections


def get_parser(config_file_name=""):
    """
    :param config_name: name of the configuration file
    :return parser: argparse parser
    """
    parser = argparse.ArgumentParser()
    # Read configuration file for defaults
    parser.add_argument('-c', '--config', type=argparse.FileType(mode='r'), default=config_file_name)
    parser.add_argument('-s1', '--setting', default='setting')
    return parser

def get_args(parser):
    """
    :param parser: argparse parser containing configuration file for default values
    :return args: arguments updated by command line inputs
    """
    args, _ = parser.parse_known_args()
    yaml_key = [args.setting]

    if args.config:
        yaml_dict = yaml.safe_load(args.config)

        set_dict = get_sum_dict(yaml_dict, yaml_key)
        parser = get_args_to_parser(set_dict, parser)
        args = parser.parse_args()
    return args

def get_args_to_parser(yaml_dict, parser):
    # Unroll what's inside the yaml
    opt_args = [['--' + key] for key, _ in yaml_dict.items()]
    opt_kwargs = [{'dest': key, 'type': type(value), 'default': value} for key, value in yaml_dict.items()]
    # Put the unrolled arguments into parser
    for p_args, p_kwargs in zip(opt_args, opt_kwargs):
        parser.add_argument(*p_args, **p_kwargs)
    return parser


def get_sum_dict(yaml_dict, yaml_key):
    """
    Get the merged dictionary from each sepearated dictionary files. 
    :yaml_dict: total_yaml_file
    """
    #assert yaml_key[0] == 'default', 'You should set the first dictionary is default setting'

    set_dict = {}
    for key in yaml_key:
        # Flatten the nested dictionary
        set_dict = dict(set_dict, **flatten_dict(yaml_dict[key]))
    return set_dict

# Adapted from https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
def flatten_dict(var_dict, parent_key='', sep='_'):
    """
    A recursive function to flatten out the dictionary.
    :param var_dict: the nested dictionary you want to flatten
    :param parent_key: if specified, use that as the header for the new key
    :param sep: what kind of separator you want to use when joining dictionary elements
    :return dict(items): flattened dictionary
    """
    items = []
    for key, value in var_dict.items():
        new_key = key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def load_parser(config_file = "config/config.yaml"):
    """Call this function to load parser"""
    parser = get_parser(config_file)
    args = get_args(parser)
    n_gpu = len(args.gpu_device_ids)
    args.distributed = n_gpu > 1
    return args