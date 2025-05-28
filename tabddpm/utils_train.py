import numpy as np
import os
import tabddpm.lib
from tabddpm.model.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: tabddpm.lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    dataset_name: str
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        X_num_t, X_cat_t, y_t = tabddpm.lib.read_pure_data(data_path, split='train')
        if X_num is not None:
            X_num['train'] = X_num_t
        if not is_y_cond:
            X_cat_t = concat_y_to_X(X_cat_t, y_t)
        if X_cat is not None:
            X_cat['train'] = X_cat_t
        y['train'] = y_t
    
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        
        X_num_t, X_cat_t, y_t = tabddpm.lib.read_pure_data(data_path, split='train')
        if not is_y_cond:
            X_num_t = concat_y_to_X(X_num_t, y_t)
        if X_num is not None:
            X_num['train'] = X_num_t
        if X_cat is not None:
            X_cat['train'] = X_cat_t
        y['train'] = y_t 

    info = tabddpm.lib.load_json(os.path.join("Data", dataset_name, 'info.json'))

    D = tabddpm.lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=tabddpm.lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = tabddpm.lib.change_val(D)
    
    return tabddpm.lib.transform_dataset(D, T, None)