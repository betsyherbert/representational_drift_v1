import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter, uniform_filter1d
from matplotlib.colors import LogNorm
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, HBox, VBox, Layout
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as opt
from tqdm import tqdm
import os
# import cmasher as cmr

plt.style.use('plots.mplstyle')