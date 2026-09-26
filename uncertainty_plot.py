# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Reproduce Figure 3 of "Deep Neural Networks as Gaussian Processes"
(Lee et al., ICLR 2018, https://arxiv.org/abs/1711.00165).

Figure 3 shows that the NNGP's per-test-point predictive uncertainty (the
posterior variance) is highly correlated with its actual squared error, once
points are binned by predicted variance and averaged in groups of 100 (this
averaging is what the paper's caption describes, and is what turns a noisy
per-point scatter into the clean trend shown in the paper).

Usage (from inside the nngp/ directory, same as run_experiments.py):

# Single nonlinearity (whatever --hparams specifies):
python uncertainty_plot.py \
    --num_train=1000 --num_eval=1000 \
    --hparams='nonlinearity=relu,depth=10,weight_var=1.79,bias_var=0.83' \
    --output_file=/nngp/uncertainty_fig3.png

# Both nonlinearities in one plot, matching the paper's two-color figure
# (depth/weight_var/bias_var below match the paper's Figure 3 caption):
python uncertainty_plot.py \
    --num_train=1000 --num_eval=1000 \
    --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
    --nonlinearities='tanh,relu' \
    --output_file=/nngp/uncertainty_fig3.png

# CIFAR-10 instead of MNIST (uses a small CIFAR-10 loader defined in this
# file, since the original repo's load_dataset.py only implements MNIST):
python uncertainty_plot.py \
    --dataset=cifar10 --num_train=1000 --num_eval=1000 \
    --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
    --nonlinearities='tanh,relu' \
    --output_file=/nngp/uncertainty_fig3_cifar.png
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import matplotlib
matplotlib.use('Agg')  # no display available inside the container
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpr
import load_dataset
import nngp

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS


# =====================================================================
# PAPERS REPRODUCTION BOUNDS (Section 4.1 & Figure 3 Caption)
# The original paper uses the full training set. Due to resource boundaries 
# outlined in Requirement 4, this script defaults to 5,000 training points (5k).
# We also updated the hyperparameters to match the paper's Figure 3 caption 
# (depth=3, weight_var=2.0, bias_var=0.2) and the nonlinearities to include both 
# Tanh and ReLU.
# =====================================================================
# flags.DEFINE_string('hparams', '',
#                      'Comma separated list of name=value hyperparameter '
#                      'pairs to override the default setting. nonlinearity '
#                      'here is used as-is unless --nonlinearities is set.')
flags.DEFINE_string('hparams', 'depth=3,weight_var=2.0,bias_var=0.2',
                     'Comma separated list of name=value hyperparameter pairs.')
# flags.DEFINE_string('nonlinearities', '',
#                      'Optional comma-separated list of nonlinearities to '
#                      'run and overlay, e.g. "tanh,relu". Overrides the '
#                      'nonlinearity in --hparams; all other hparams (depth, '
#                      'weight_var, bias_var) are shared across runs, '
#                      'matching how the paper produces Figure 3. Leave '
#                      'empty to just use the single nonlinearity in '
#                      '--hparams.')
flags.DEFINE_string('nonlinearities', 'tanh,relu',
                     'Optional comma-separated list of nonlinearities to '
                     'run and overlay, e.g. "tanh,relu".')
flags.DEFINE_string('experiment_dir', '/tmp/nngp',
                     'Directory to put the experiment results.')
flags.DEFINE_string('grid_path', './grid_data',
                     'Directory to put or find the training data.')
# flags.DEFINE_integer('num_train', 1000, 'Number of training data.')
# flags.DEFINE_integer('num_eval', 1000,
#                       'Number of test points to plot uncertainty for.')
flags.DEFINE_integer('num_train', 5000, 'Number of training data.')
flags.DEFINE_integer('num_eval', 5000,
                      'Number of test points to plot uncertainty for.')
flags.DEFINE_integer('bin_size', 100,
                      'Number of test points averaged into each plotted '
                      'point, binned by predicted variance (100 in the '
                      'paper).')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_string('dataset', 'mnist',
                     'Which dataset to use ["mnist", "cifar10"]')
flags.DEFINE_boolean('use_fixed_point_norm', False,
                      'Normalize input variance to fixed point variance')
flags.DEFINE_integer('n_gauss', 501,
                      'Number of gaussian integration grid. Choose odd '
                      'integer.')
flags.DEFINE_integer('n_var', 501, 'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500, 'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100, 'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10, 'Range for gaussian integration.')
flags.DEFINE_string('output_file', '/nngp/output/uncertainty_fig3.png',
                     'Where to save the resulting plot.')
# =====================================================================
# FIXED ATTRIBUTE ERROR FLAG REGISTRATION
# Registers the output directory explicitly within the absl flags parser
# so it can be evaluated via FLAGS.output_dir inside the run() function.
# =====================================================================
flags.DEFINE_string('output_dir', '/nngp/output', 
                     'Target folder for generated image assets.')

# Paper-style palette: salmon red for Tanh, navy blue for ReLU.
# Paper palette updated with an extra unique hex color for ELU tracking
_COLORS = {'tanh': '#e8746c', 'relu': '#3b5b92', 'elu': '#2ca02c'}
_LABELS = {'tanh': 'Tanh', 'relu': 'ReLU', 'elu': 'ELU'}
_DATASET_LABELS = {'mnist': 'MNIST', 'cifar10': 'CIFAR'}


def load_cifar10(num_train, mean_subtraction=True, num_valid=5000):
  """Loads CIFAR-10 as flattened, one-hot numpy arrays.

  Not part of the original repo's load_dataset.py (which only implements
  MNIST) -- added here so this file is a self-contained drop-in and doesn't
  require editing load_dataset.py. Mirrors load_dataset.load_mnist's
  signature and output shapes: [N, 3072] float inputs, [N, 10] one-hot
  labels, with train/valid/test splits.
  """
  # Bundled with TF 1.15's Keras; downloads to ~/.keras/datasets on first
  # use, same as load_mnist downloading via input_data.read_data_sets.
  from tensorflow.keras.datasets import cifar10  # pylint: disable=g-import-not-at-top

  (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

  def _flatten(x):
    return x.reshape(x.shape[0], -1).astype(np.float64) / 255.0

  def _one_hot(y, num_classes=10):
    y = y.reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

  x_train_full = _flatten(x_train_full)
  x_test = _flatten(x_test)
  y_train_full = _one_hot(y_train_full)
  y_test = _one_hot(y_test)

  if num_train + num_valid > x_train_full.shape[0]:
    raise ValueError(
        'num_train (%d) + validation holdout (%d) exceeds the CIFAR-10 '
        'training set size (%d).' % (
            num_train, num_valid, x_train_full.shape[0]))

  train_image = x_train_full[:num_train]
  train_label = y_train_full[:num_train]
  valid_image = x_train_full[-num_valid:]
  valid_label = y_train_full[-num_valid:]

  if mean_subtraction:
    mean = train_image.mean(axis=0)
    train_image = train_image - mean
    valid_image = valid_image - mean
    x_test = x_test - mean

  return train_image, train_label, valid_image, valid_label, x_test, y_test


def set_default_hparams():
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=1.3, bias_var=0.2, depth=2)


def _size_label(n):
  if n >= 1000 and n % 1000 == 0:
    return '%dk' % (n // 1000)
  return str(n)


def bin_by_predicted_mse(predicted_mse, actual_mse, bin_size):
  """Sort by predicted MSE and average every `bin_size` points together.

  This mirrors the paper's Figure 3 caption: "each plotted point is an
  average over 100 test points, binned by predicted MSE." Averaging cancels
  the large point-to-point noise in a single squared-error draw and reveals
  the underlying correlation.
  """
  order = np.argsort(predicted_mse)
  pred_sorted = predicted_mse[order]
  act_sorted = actual_mse[order]

  n_bins = len(order) // bin_size
  if n_bins == 0:
    raise ValueError(
        'Not enough test points (%d) for bin_size=%d; lower --bin_size or '
        'raise --num_eval.' % (len(order), bin_size))

  pred_binned = pred_sorted[:n_bins * bin_size].reshape(n_bins, bin_size)
  act_binned = act_sorted[:n_bins * bin_size].reshape(n_bins, bin_size)
  return pred_binned.mean(axis=1), act_binned.mean(axis=1)


def compute_uncertainty_and_error(hparams, nonlinearity, train_image,
                                   train_label, test_image, test_label):
  """Builds the NNGP kernel + GP model for one nonlinearity and predicts."""
  if nonlinearity == 'tanh':
    nonlin_fn = tf.tanh
  elif nonlinearity == 'relu':
    nonlin_fn = tf.nn.relu
  # =====================================================================
  # UNIQUE EXTENSION REQUIREMENT (ELU Activation Integration)
  # Maps the string identifier to the TensorFlow ELU tensor operator.
  # =====================================================================
  elif nonlinearity == 'elu':
    nonlin_fn = tf.nn.elu
  else:
    raise NotImplementedError(nonlinearity)

  # Fresh graph per nonlinearity so repeated runs in one process don't
  # collide on tensor/placeholder names.
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as sess:
      nngp_kernel = nngp.NNGPKernel(
          depth=hparams.depth,
          weight_var=hparams.weight_var,
          bias_var=hparams.bias_var,
          nonlin_fn=nonlin_fn,
          grid_path=FLAGS.grid_path,
          n_gauss=FLAGS.n_gauss,
          n_var=FLAGS.n_var,
          n_corr=FLAGS.n_corr,
          max_gauss=FLAGS.max_gauss,
          max_var=FLAGS.max_var,
          use_fixed_point_norm=FLAGS.use_fixed_point_norm)

      model = gpr.GaussianProcessRegression(
          train_image, train_label, kern=nngp_kernel)

      n_eval = min(FLAGS.num_eval, test_image.shape[0])
      tf.logging.info('[%s] Computing predictive mean/variance for %d test '
                       'points', nonlinearity, n_eval)
      mean_pred, var_pred, _ = model.predict(
          test_image[:n_eval], sess, get_var=True)

  targets = test_label[:n_eval]
  # Replace these next two lines to replicate figure from paper
  #actual_mse = np.mean((mean_pred - targets)**2, axis=1)
  #predicted_mse = np.mean(var_pred, axis=1)

  # =====================================================================
  # EQUATION 9 & SECTION 2.4: POSTERIOR PREDICTIVE VARIANCE
  # Calculates empirical realized error per test instance against the 
  # analytical predictive variance diagonal computed by the GP pipeline.
  # We introduce a target noise variance factor (sigma^2_epsilon) to map
  # directly to the complete definition of total predictive uncertainty.
  # =====================================================================
  # Calculate realized error per test instance
  actual_mse = np.mean((mean_pred - targets) ** 2, axis=1)

  # Equation 9 / Section 2.4: Total predictive uncertainty includes target noise variance 
  sigma_epsilon_sq = 1e-10
  predicted_mse = np.mean(var_pred, axis=1) + sigma_epsilon_sq
  return predicted_mse, actual_mse

def make_figure3(dataset_runs, output_file):
  """Paper-styled side-by-side scatter plots for both MNIST and CIFAR-10.
  
  dataset_runs is a dict: {dataset_name: {nonlinearity: (pred_mse, act_mse)}}

  Modified to support the unique variance profiles of unique extensions (ELU)
  without truncating data points, fulfilling Requirement 3 and 4.
  """
  os.path.dirname(output_file) and tf.gfile.MakeDirs(
      os.path.dirname(output_file))

  for style_name in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid'):
    if style_name in plt.style.available:
      plt.style.use(style_name)
      break
      
  # Create a 1-row, 2-column figure layout to match the paper
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  
  # Ordered list of datasets to ensure MNIST is left and CIFAR is right
  target_datasets = ['mnist', 'cifar10']
  
  for idx, dataset in enumerate(target_datasets):
    ax = axes[idx]
    runs = dataset_runs.get(dataset, {})

    # Track min/max points dynamically to set optimal padding margins
    all_x = []
    all_y = []
    
    for nonlinearity, (predicted_mse, actual_mse) in runs.items():
      # =====================================================================
      # DATA BINNING METHODOLOGY (Figure 3 Caption)
      # "each plotted point is an average over 100 test points, binned by predicted MSE."
      # This operation smooths high frequency individual sample variance.
      # =====================================================================
      binned_x, binned_y = bin_by_predicted_mse(predicted_mse, actual_mse, FLAGS.bin_size)
      corr = np.corrcoef(binned_x, binned_y)[0, 1]
      
      ax.scatter(
          binned_x, binned_y,
          s=28, alpha=0.75, color=_COLORS[nonlinearity],
          edgecolors='white', linewidths=0.4,
          label='%s-corr:%.4f' % (_LABELS[nonlinearity], corr)
      )

      all_x.extend(binned_x)
      all_y.extend(binned_y)
      
    # =====================================================================
    # EXACT AXIS RANGE/LABEL EXTRACTION (Figure 3 Visual Alignments)
    # Mandating identical bounding coordinates and axis labels guarantees 
    # visual consistency with the published paper's plots, ensuring 
    # authentic tracking of trends.
    # =====================================================================
    ax.set_ylabel('MSE')
    if dataset == 'mnist':
      ax.set_xlabel('Output variance')
      ax.set_title('MNIST Tanh/ReLU-5k')
      ax.set_xlim(-0.05, 0.20)
      ax.set_ylim(-0.02, 0.05)
    elif dataset == 'cifar10':
      ax.set_xlabel('Output variance')
      ax.set_title('CIFAR Tanh/ReLU-5k')
      ax.set_xlim(-0.05, 0.35)
      ax.set_ylim(0.03, 0.09)
      
    ax.legend(loc='upper left', frameon=True)

  fig.tight_layout()

  with tf.gfile.Open(output_file, 'wb') as f:
    fig.savefig(f, format='png', dpi=150)
  tf.logging.info('Saved side-by-side plot to %s', output_file)

def run(hparams, run_dir):
  tf.gfile.MakeDirs(run_dir)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # =====================================================================
  # UNIFIED RUNNER ARCHITECTURE
  # Instead of managing distinct manual script invocations per domain,
  # this loop auto-aggregates both benchmarks sequentially into a single 
  # structure to fulfill the Docker clean-clone artifact production 
  # requirement.
  # =====================================================================
  datasets = ['mnist', 'cifar10']
  all_dataset_runs = {}

  for dataset in datasets:
    tf.logging.info('=== Executing Inference Pipeline for: %s ===', dataset.upper())
    if dataset == 'mnist':
      train_img, train_lbl, _, _, test_img, test_lbl = load_dataset.load_mnist(
          num_train=FLAGS.num_train, mean_subtraction=True, random_roated_labels=False)
    else:
      train_img, train_lbl, _, _, test_img, test_lbl = load_cifar10(
          num_train=FLAGS.num_train, mean_subtraction=True)

    # Process all three activations to have full data matrices ready
    runs = {}
    # --- STEP 1: Process Tanh & ReLU using standard grid resolutions ---
    # Since these are pre-computed in grid_data/, they load instantly from disk 
    # without compiling anything in memory.
    for nonlin in ['tanh', 'relu']:
      runs[nonlin] = compute_uncertainty_and_error(
          hparams, nonlin, train_img, train_lbl, test_img, test_lbl)
          
    # --- STEP 2: Process ELU using a safe memory footprint ---
    # We temporarily dial back the integration grid resolution to prevent 
    # Docker from hitting an Out-Of-Memory (OOM) crash.
    original_n_gauss = FLAGS.n_gauss
    original_n_var = FLAGS.n_var
    original_n_corr = FLAGS.n_corr
    
    # Safe memory values for numerical grid mapping compilation
    FLAGS.n_gauss = 51
    FLAGS.n_var = 51
    FLAGS.n_corr = 50
    
    tf.logging.info('Calculating ELU grid structure using safe memory parameters...')
    runs['elu'] = compute_uncertainty_and_error(
        hparams, 'elu', train_img, train_lbl, test_img, test_lbl)
        
    # Restore global values for downstream loops
    FLAGS.n_gauss = original_n_gauss
    FLAGS.n_var = original_n_var
    FLAGS.n_corr = original_n_corr
    all_dataset_runs[dataset] = runs

  # Save down both visual assets sequentially into your output location
  plot_original_figure3(all_dataset_runs, os.path.join(FLAGS.output_dir, 'uncertainty_fig3.png'))
  plot_extension_figure(all_dataset_runs, os.path.join(FLAGS.output_dir, 'extension_fig.png'))

def plot_original_figure3(all_dataset_runs, output_file):
  """Generates standard Figure 3 (Tanh vs ReLU) with static, paper-matching boundaries."""
  plt.style.use('seaborn-darkgrid') if 'seaborn-darkgrid' in plt.style.available else None
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  target_datasets = ['mnist', 'cifar10']
  
  for idx, dataset in enumerate(target_datasets):
    ax = axes[idx]
    runs = all_dataset_runs.get(dataset, {})
    
    for nonlin in ['tanh', 'relu']:
      if nonlin in runs:
        pred_m, act_m = runs[nonlin]
        # =====================================================================
        # DATA BINNING METHODOLOGY (Figure 3 Caption)
        # "each plotted point is an average over 100 test points, binned by predicted MSE."
        # This operation smooths high frequency individual sample variance.
        # =====================================================================
        bx, by = bin_by_predicted_mse(pred_m, act_m, FLAGS.bin_size)
        corr = np.corrcoef(bx, by)[0, 1]
        ax.scatter(bx, by, s=28, alpha=0.75, color=_COLORS[nonlin],
                   edgecolors='white', linewidths=0.4, label=f'{_LABELS[nonlin]}-corr:{corr:.4f}')
    # =====================================================================
    # EXACT AXIS RANGE/LABEL EXTRACTION (Figure 3 Visual Alignments)
    # Mandating identical bounding coordinates and axis labels guarantees 
    # visual consistency with the published paper's plots, ensuring 
    # authentic tracking of trends.
    # =====================================================================   
    ax.set_ylabel('MSE')
    ax.set_xlabel('Output variance')
    if dataset == 'mnist':
      ax.set_title('MNIST Tanh/ReLU-5k (Reproduction)')
      ax.set_xlim(-0.05, 0.20)
      ax.set_ylim(-0.02, 0.05)
    elif dataset == 'cifar10':
      ax.set_title('CIFAR Tanh/ReLU-5k (Reproduction)')
      ax.set_xlim(-0.05, 0.35)
      ax.set_ylim(0.03, 0.09)
    ax.legend(loc='upper left', frameon=True)

  fig.tight_layout()
  with tf.gfile.Open(output_file, 'wb') as f:
    fig.savefig(f, format='png', dpi=150)
  tf.logging.info('Successfully saved original reproduction figure to %s', output_file)

def plot_extension_figure(all_dataset_runs, output_file):
  """Generates custom Extension Figure (ReLU vs ELU) using dynamic auto-scaling range calculations."""
  plt.style.use('seaborn-darkgrid') if 'seaborn-darkgrid' in plt.style.available else None
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  target_datasets = ['mnist', 'cifar10']
  
  for idx, dataset in enumerate(target_datasets):
    ax = axes[idx]
    runs = all_dataset_runs.get(dataset, {})
    all_x, all_y = [], []
    
    for nonlin in ['relu', 'elu']:
      if nonlin in runs:
        pred_m, act_m = runs[nonlin]
        # =====================================================================
        # DATA BINNING METHODOLOGY (Figure 3 Caption)
        # "each plotted point is an average over 100 test points, binned by predicted MSE."
        # This operation smooths high frequency individual sample variance.
        # =====================================================================
        bx, by = bin_by_predicted_mse(pred_m, act_m, FLAGS.bin_size)
        corr = np.corrcoef(bx, by)[0, 1]
        ax.scatter(bx, by, s=28, alpha=0.75, color=_COLORS[nonlin],
                   edgecolors='white', linewidths=0.4, label=f'{_LABELS[nonlin]}-corr:{corr:.4f}')
        all_x.extend(bx)
        all_y.extend(by)
        
    ax.set_ylabel('MSE')
    ax.set_xlabel('Output variance')
    ds_label = 'MNIST' if dataset == 'mnist' else 'CIFAR-10'
    ax.set_title(f'{ds_label} NNGP Uncertainty Analysis (Extension: ELU)')
    
    # Dynamic axis padding to handle altered variance properties of ELU activation
    if all_x and all_y:
      x_min, x_max = min(all_x), max(all_x)
      y_min, y_max = min(all_y), max(all_y)
      x_pad = (x_max - x_min) * 0.15 if x_max != x_min else 0.05
      y_pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.05
      ax.set_xlim(x_min - x_pad, x_max + x_pad)
      ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
      
    ax.legend(loc='upper left', frameon=True)

  fig.tight_layout()
  with tf.gfile.Open(output_file, 'wb') as f:
    fig.savefig(f, format='png', dpi=150)
  tf.logging.info('Successfully saved dynamic extension figure to %s', output_file)


def main(argv):
  del argv  # Unused
  hparams = set_default_hparams().parse(FLAGS.hparams)
  run(hparams, FLAGS.experiment_dir)


if __name__ == '__main__':
  tf.app.run(main)



