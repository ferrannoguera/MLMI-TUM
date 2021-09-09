import torchvision
import os
import shutil
import numpy as np


def get_model_file(model_path):
    model_files = [m for m in os.listdir(model_path) if "best_model" in m ]
    assert len(model_files) == 1, f"Either no model in {model_path} or more than one 'best_model'." 
    return os.path.join(model_path, model_files[0])
    

def beautify_tensorboard(x):
    return torchvision.utils.make_grid(x.float(), normalize=True)

def beautify_tensorboard_oct(*args, **kwargs):
    std_ = 53.9434
    mean_ = 46.3758
    return beautify_tensorboard_rescale(*args, std_=std_, mean_=mean_, **kwargs)

def beautify_tensorboard_oct_ulbl(*args, **kwargs):
    std_ = 0.1762
    mean_ = 0.25
    return beautify_tensorboard_rescale(*args, std_=std_, mean_=mean_, **kwargs)

def beautify_tensorboard_rescale(x, is_mask=False, std_=1., mean_=0.):
    return torchvision.utils.make_grid(x.float(), normalize=True) if is_mask else \
        torchvision.utils.make_grid(x.float() * std_ + mean_, normalize=True)

def get_next_run(log_path):
    run = [int(e.split("_")[1]) for e in os.listdir(log_path) if os.path.isdir(os.path.join(log_path,e)) and e.startswith("run_")]
    return max(run) + 1 if len(run) > 0 else 0

def ensure_abs_path(path, file_path):
    """
    ensure path is abspath - if relative path resolve to abs path relative to file path
    """
    return os.path.join(file_path,path) if not os.path.isabs(path) else path

def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith(ext)]



def random_sample(files, num):
    return list(np.random.permutation(files)[0:num])

def move_files(files, source_path, dest_path):
    if not os.path.isdir(source_path) or not os.path.isdir(dest_path):
        raise ValueError(f"Source path: {source_path} or dest_path: {dest_path} do not exist")
    source_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path,f))]
    for f in files:
        if not f in source_files:
            raise ValueError(f"File {f} not in source directory {source_path}")
    for f in files:
        shutil.move(os.path.join(source_path, f), os.path.join(dest_path,f))

