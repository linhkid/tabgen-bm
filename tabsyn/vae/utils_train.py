import numpy as np
import os
import tabsyn.src as src
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = None
    T_dict['num_nan_policy'] = None
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = None
    T_dict['y_policy'] = None

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = (dataset.num_transform.inverse_transform if dataset.num_transform else None)
            cat_inverse = (dataset.cat_transform.inverse_transform if dataset.cat_transform else None)
            #num_inverse = dataset.num_transform.inverse_transform
            #cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t

            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    # Try to find info.json in multiple locations
    try:
        # First try data_path directory
        info = src.load_json(os.path.join(data_path, 'info.json'))
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            # Then try parent directory
            info = src.load_json(os.path.join(os.path.dirname(data_path), 'info.json'))
        except (FileNotFoundError, json.JSONDecodeError):
            try:
                # Then try Data/dataset directory (extract dataset name from path)
                dataset_name = os.path.basename(data_path.rstrip('/'))
                if 'seed' in dataset_name:  # If path ends with seed dir, get parent
                    dataset_name = os.path.basename(os.path.dirname(data_path.rstrip('/')))
                info = src.load_json(os.path.join('Data', dataset_name, 'info.json'))
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a default info if all attempts fail
                print(f"Warning: info.json not found. Creating a default version.")
                # Determine if binary or multiclass from y_train data
                y_train_path = os.path.join(data_path, 'y_train.npy')
                if os.path.exists(y_train_path):
                    y_train = np.load(y_train_path)
                    unique_classes = len(np.unique(y_train))
                    task_type = 'multiclass' if unique_classes > 2 else 'binclass'
                else:
                    # Default to multiclass if can't determine
                    task_type = 'multiclass'
                
                # Create and save a basic info.json
                info = {'task_type': task_type}
                try:
                    with open(os.path.join(data_path, 'info.json'), 'w') as f:
                        import json
                        json.dump(info, f, indent=2)
                    print(f"Created default info.json with task_type: {task_type}")
                except Exception as e:
                    print(f"Note: Could not save info.json: {e}, but will continue with generated info")

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)