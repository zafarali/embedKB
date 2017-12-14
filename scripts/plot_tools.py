import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import seaborn as sns
import pandas as pd
from scipy.interpolate import BSpline
sns.set_style('white')
sns.set_context('paper', font_scale=2)
MODELS = ['Bilinear', 'TransE', 'NeuralTensorNetwork', 'SingleLayer', 'StructuredEmbedding']
COLORS = sns.color_palette(n_colors=len(MODELS))
get_files = lambda model_name, dset, root: glob(os.path.join(root, 'run_'+model_name+'_checkpoints_'+dset+'__summaries_*,tag_score_loss.csv'))

def load_file_locations(path='./'):
    """
    Loads files
    """
    csv_locations = {}
    for model_name in MODELS:
        csv_locations[model_name] = get_files(model_name, '*', path)        
    return csv_locations

def load_curves(locations):
    """
    Loads the curves
    """
    loaded_curves = {}
    for model_name, files in locations.items():
        loaded_curves[model_name] = {}
        for file_name in files:
            if 'val' in file_name:
                loaded_curves[model_name]['val'] = pd.read_csv(file_name)['Value'].values
            else:
                loaded_curves[model_name]['train'] = pd.read_csv(file_name)['Value'].values
                
    return loaded_curves


def plot_curves(loaded_curves, ax=None):
    if not ax:
        f = plt.figure()
        ax = f.add_subplot(111)
    for i, (model_name, curves) in enumerate(loaded_curves.items()):
        curve = curves['val']
        xnew = np.linspace(0,99,100) #300 represents number of points to make between T.min and T.max
        spl = BSpline(np.arange(0, 100), curve, 2)
        smooth_curve = spl(xnew)
#         ax.plot(curves['train'], c=COLORS[i], linestyle='--', label=model_name)
        ax.plot(smooth_curve, c=COLORS[i], linestyle='-', label=model_name)        
    return ax



if __name__ == '__main__':
    f = plt.figure(figsize=(12, 5))
    ax = f.add_subplot(111)
    plot_curves(
        load_curves(
            load_file_locations('./')
            ),
        ax=ax)

    ax.legend()
    ax.set_title('Validation Curves')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Objective')
    sns.despine()
    plt.savefig('./valcurves.pdf')