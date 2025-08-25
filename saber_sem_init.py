import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import math
import numpy as np
import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import arviz as az
import bambi as bmb
import catboost as cb
from pygam import LinearGAM, s, te, l

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score, make_scorer, mean_squared_error, r2_score

from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

import optuna
from optuna.samplers import TPESampler


COLOR_DICT = {
    "BLUE": "#2a5674",
    "RED": "#b13f64",
    "TEAL": "#4a9b8f",
    "CORAL": "#f47c65",
    "PURPLE": "#6a4a99",
    "ORANGE": "#f28c28",
    "GREEN": "#4a9b4a",
    "PINK": "#d47ca6",
    "MAROON": "#8b2439",
    "NAVY": "#1a237e",
    "LIME": "#b6e880",
    "CYAN": "#00bcd4",
    "BROWN": "#795548",
    "MAGENTA": "#e040fb",
    "GOLD": "#e6b54a",
}

colors = [COLOR_DICT[color] for color in COLOR_DICT.keys()]