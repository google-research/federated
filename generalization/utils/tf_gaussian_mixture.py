# pytype: skip-file

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# Modified by Google LLC
# License: BSD 3 clause
"""Scikit-learn Gaussian Mixture Model rewritten in TF."""

import abc
import math
import time
from typing import Optional
import warnings

from absl import logging
import numpy as np
import sklearn.cluster

import tensorflow as tf

# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield

DTYPE = tf.float64


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
  """Check the input data X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, shape (n_samples, n_features)
  """
  X = tf.cast(X, DTYPE)

  if ensure_min_samples > 0:
    n_samples = X.shape[0]
    if n_samples < ensure_min_samples:
      raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                       " minimum of %d is required." %
                       (n_samples, X.shape, ensure_min_samples))

  if n_components is not None and X.shape[0] < n_components:
    raise ValueError("Expected n_samples >= n_components "
                     "but got n_components = %d, n_samples = %d" %
                     (n_components, X.shape[0]))
  if n_features is not None and X.shape[1] != n_features:
    raise ValueError("Expected the input data X have %d features, "
                     "but got %d features" % (n_features, X.shape[1]))
  return X


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
  """Estimate the full covariance matrices.

  Args:
    resp : array-like of shape (n_samples, n_components)
    X : array-like of shape (n_samples, n_features)
    nk : array-like of shape (n_components,)
    means : array-like of shape (n_components, n_features)
    reg_covar : float

  Returns:
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
  """
  n_components, n_features = means.shape

  covariances_list = [None for _ in range(n_components)]
  regularizer = tf.eye(n_features, dtype=DTYPE) * reg_covar

  for k in range(n_components):
    diff = X - means[k]
    covariances_list[k] = tf.matmul(
        tf.expand_dims(resp[:, k], 1) * diff, diff,
        transpose_a=True) / nk[k] + regularizer

  covariances = tf.stack(covariances_list)
  return covariances


def _estimate_gaussian_parameters(X, resp, reg_covar, weights, means,
                                  covariances):
  """Estimate the Gaussian distribution parameters.

  Args:
    X : array-like of shape (n_samples, n_features) The input data array.
    resp : array-like of shape (n_samples, n_components) The responsibilities
      for each data sample in X.
    reg_covar : float The regularization added to the diagonal of the covariance
      matrices.
  Assigns to the following tf.variable:
    weights : array-like of shape (n_components,) The numbers of data samples in
      the current components.
    means : array-like of shape (n_components, n_features) The centers of the
      current components.
    covariances : array-like The covariance matrix of the current components.
  """
  nk = tf.reduce_sum(
      resp, axis=0) + 10 * np.finfo(resp.dtype.as_numpy_dtype).eps
  weights.assign(nk / X.shape[0])
  means.assign(tf.matmul(resp, X, transpose_a=True) / nk[:, tf.newaxis])
  covariances.assign(
      _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar))


def _compute_precision_cholesky(covariances, precisions_cholesky):
  """Compute the Cholesky decomposition of the precisions.

  Args:
    covariances : array-like The covariance matrix of the current components.
  Assigns:
    precisions_cholesky : array-like The cholesky decomposition of sample
      precisions of the current components.
  """
  # estimate_precision_error_message = (
  #     "Fitting the mixture model failed because some components have "
  #     "ill-defined empirical covariance (for instance caused by singleton "
  #     "or collapsed samples). Try to decrease the number of components, "
  #     "or increase reg_covar.")

  n_features = covariances.shape[1]

  cov_chols = tf.linalg.cholesky(covariances)
  precisions_cholesky.assign(
      tf.linalg.triangular_solve(
          cov_chols, tf.eye(n_features, dtype=DTYPE), lower=True))


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol):
  """Compute the log-det of the cholesky decomposition of matrices.

  Args:
    matrix_chol : array-like Cholesky decompositions of the matrices. shape of
      (n_components, n_features, n_features)

  Returns:
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
  """
  log_det_chol = (
      tf.reduce_sum(tf.math.log(tf.linalg.diag_part(matrix_chol)), 1))

  return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol):
  """Estimate the log Gaussian probability.

  Args:
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like Cholesky decompositions of the precision
      matrices. shape of (n_components, n_features, n_features)

  Returns:
    log_prob : array, shape (n_samples, n_components)
  """
  _, n_features = X.shape
  n_components, _ = means.shape
  # det(precision_chol) is half of det(precision)
  log_det = _compute_log_det_cholesky(precisions_chol)

  log_probs = [None for _ in range(n_components)]

  # This loop is known to cause GPU OOM for large-scale experiments.
  for k in range(means.shape[0]):
    mu = means[k, :]
    prec_chol = precisions_chol[k, :, :]
    y = tf.matmul(
        X, prec_chol, transpose_b=True) - tf.linalg.matvec(prec_chol, mu)
    log_probs[k] = tf.reduce_sum(tf.square(y), axis=1)

  log_prob = tf.stack(log_probs, axis=1)

  return -.5 * (n_features * tf.math.log(2 * tf.constant(math.pi, dtype=DTYPE))
                + log_prob) + log_det


class GaussianMixture(
    sklearn.base.DensityMixin,
    sklearn.base.BaseEstimator,
    metaclass=abc.ABCMeta):
  """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    kmeans_batch_size : int or None, default=10000:
        Batch size for kmeans if init_params is 'kmeans'. If None, full batch
        kmeans will be used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        shape: (n_components, n_features, n_features)

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        Shape: (n_components, n_features, n_features)

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. Shape: (n_components, n_features, n_features)

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

  """

  def __init__(self,
               n_components=1,
               *,
               tol=1e-3,
               reg_covar=1e-6,
               max_iter=100,
               n_init=1,
               init_params="kmeans",
               kmeans_batch_size: Optional[int] = None,
               random_state=None,
               warm_start=False,
               verbose=0,
               verbose_interval=10):
    self.n_components = n_components
    self.tol = tol
    self.reg_covar = reg_covar
    self.max_iter = max_iter
    self.n_init = n_init
    self.init_params = init_params
    self.kmeans_batch_size = kmeans_batch_size
    self.random_state = random_state
    self.warm_start = warm_start
    self.verbose = verbose
    self.verbose_interval = verbose_interval

  def _check_initial_parameters(self, X):
    """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
    """
    if self.n_components < 1:
      raise ValueError("Invalid value for 'n_components': %d "
                       "Estimation requires at least one component" %
                       self.n_components)

    if self.tol < 0.:
      raise ValueError("Invalid value for 'tol': %.5f "
                       "Tolerance used by the EM must be non-negative" %
                       self.tol)

    if self.n_init < 1:
      raise ValueError("Invalid value for 'n_init': %d "
                       "Estimation requires at least one run" % self.n_init)

    if self.max_iter < 1:
      raise ValueError("Invalid value for 'max_iter': %d "
                       "Estimation requires at least one iteration" %
                       self.max_iter)

    if self.reg_covar < 0.:
      raise ValueError("Invalid value for 'reg_covar': %.5f "
                       "regularization on covariance must be "
                       "non-negative" % self.reg_covar)

  def _initialize_parameters(self, X, random_state):
    """Initialize the model parameters.

    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)

    random_state : RandomState
        A random number generator instance that controls the random seed
        used for the method chosen to initialize the parameters.
    """
    n_samples, _ = X.shape

    if self.init_params == "kmeans":
      resp = np.zeros((n_samples, self.n_components))
      if self.kmeans_batch_size is not None and self.kmeans_batch_size < n_samples:
        label = sklearn.cluster.MiniBatchKMeans(
            n_clusters=self.n_components,
            n_init=1,
            batch_size=self.kmeans_batch_size,
            random_state=random_state,
            verbose=self.verbose).fit(X).labels_
      else:
        label = sklearn.cluster.KMeans(
            n_clusters=self.n_components,
            n_init=1,
            random_state=random_state,
            verbose=self.verbose).fit(X).labels_
      resp[np.arange(n_samples), label] = 1
    elif self.init_params == "random":
      resp = random_state.rand(n_samples, self.n_components)
      resp /= tf.reduce_sum(resp, axis=1)[:, np.newaxis]
    else:
      raise ValueError("Unimplemented initialization method '%s'" %
                       self.init_params)

    resp = tf.cast(resp, DTYPE)
    self._initialize(X, resp)

  def _initialize(self, X, resp):
    """Initialization of the Gaussian mixture parameters.

    Args:
      X : array-like of shape (n_samples, n_features)
      resp : array-like of shape (n_samples, n_components)
    """
    n_features = X.shape[1]
    n_components = resp.shape[1]

    self.weights_ = tf.Variable(
        tf.random.normal((n_components,), dtype=tf.float64))
    self.means_ = tf.Variable(
        tf.random.normal((n_components, n_features), dtype=tf.float64))
    self.covariances_ = tf.Variable(
        tf.random.normal((n_components, n_features, n_features),
                         dtype=tf.float64))
    self.precisions_cholesky_ = tf.Variable(
        tf.random.normal((n_components, n_features, n_features),
                         dtype=tf.float64))

    self._best_weights_ = tf.Variable(
        tf.random.normal((n_components,), dtype=tf.float64))
    self._best_means_ = tf.Variable(
        tf.random.normal((n_components, n_features), dtype=tf.float64))
    self._best_covariances_ = tf.Variable(
        tf.random.normal((n_components, n_features, n_features),
                         dtype=tf.float64))
    self._best_precisions_cholesky_ = tf.Variable(
        tf.random.normal((n_components, n_features, n_features),
                         dtype=tf.float64))

    _estimate_gaussian_parameters(X, resp, self.reg_covar, self.weights_,
                                  self.means_, self.covariances_)

    _compute_precision_cholesky(self.covariances_, self.precisions_cholesky_)

  def fit(self, X, y=None):
    """Estimate model parameters with the EM algorithm.

    The method fits the model ``n_init`` times and sets the parameters with
    which the model has the largest likelihood or lower bound. Within each
    trial, the method iterates between E-step and M-step for ``max_iter``
    times until the change of likelihood or lower bound is less than
    ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
    If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
    initialization is performed upon the first call. Upon consecutive
    calls, training starts where it left off.

    Args:
      X : array-like of shape (n_samples, n_features) List of
        n_features-dimensional data points. Each row corresponds to a single
        data point.

    Returns:
      self
    """
    self.fit_predict(X, y)
    return self

  def fit_predict(self, X, y=None):
    """Estimate model parameters using X and predict the labels for X.

    The method fits the model n_init times and sets the parameters with
    which the model has the largest likelihood or lower bound. Within each
    trial, the method iterates between E-step and M-step for `max_iter`
    times until the change of likelihood or lower bound is less than
    `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
    raised. After fitting, it predicts the most probable label for the
    input data points.

    .. versionadded:: 0.20

    Args:
      X : array-like of shape (n_samples, n_features) List of
        n_features-dimensional data points. Each row corresponds to a single
        data point.

    Returns:
      labels : array, shape (n_samples,)
        Component labels.
    """
    X = _check_X(X, self.n_components, ensure_min_samples=2)
    self._check_initial_parameters(X)

    # if we enable warm_start, we will have a unique initialisation
    do_init = not (self.warm_start and hasattr(self, "converged_"))
    n_init = self.n_init if do_init else 1

    max_lower_bound = -np.infty
    self.converged_ = False

    random_state = sklearn.utils.check_random_state(self.random_state)

    for init in range(n_init):
      self._print_verbose_msg_init_beg(init)

      if do_init:
        self._initialize_parameters(X, random_state)

      lower_bound = (-np.infty if do_init else self.lower_bound_)

      for n_iter in range(1, self.max_iter + 1):
        prev_lower_bound = lower_bound

        log_prob_norm, log_resp = self._e_step(X)
        self._m_step(X, log_resp)
        lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

        change = lower_bound - prev_lower_bound
        self._print_verbose_msg_iter_end(n_iter, change)

        if abs(change) < self.tol:
          self.converged_ = True
          break

      self._print_verbose_msg_init_end(lower_bound)

      if lower_bound > max_lower_bound:
        max_lower_bound = lower_bound
        self._best_weights_.assign(self.weights_)
        self._best_means_.assign(self.means_)
        self._best_covariances_.assign(self.covariances_)
        self._best_precisions_cholesky_.assign(self.precisions_cholesky_)
        best_n_iter = n_iter

    if not self.converged_:
      warnings.warn(
          "Initialization %d did not converge. "
          "Try different init parameters, "
          "or increase max_iter, tol "
          "or check for degenerate data." % (init + 1),
          sklearn.exceptions.ConvergenceWarning)

    self.weights_.assign(self._best_weights_)
    self.means_.assign(self._best_means_)
    self.covariances_.assign(self._best_covariances_)
    self.precisions_cholesky_.assign(self._best_precisions_cholesky_)

    self.n_iter_ = best_n_iter
    self.lower_bound_ = max_lower_bound

    # Always do a final e-step to guarantee that the labels returned by
    # fit_predict(X) are always consistent with fit(X).predict(X)
    # for any value of max_iter and tol (and any random_state).
    _, log_resp = self._e_step(X)

    return tf.math.argmax(log_resp, axis=1)

  def _e_step(self, X):
    """E step.

    Args:
      X : array-like of shape (n_samples, n_features)

    Returns:
      log_prob_norm : float
        Mean of the logarithms of the probabilities of each sample in X

      log_responsibility : array, shape (n_samples, n_components)
        Logarithm of the posterior probabilities (or responsibilities) of
        the point of each sample in X.
    """
    log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
    return tf.reduce_mean(log_prob_norm), log_resp

  def _m_step(self, X, log_resp):
    """M step.

    Args:
      X : array-like of shape (n_samples, n_features)
      log_resp : array-like of shape (n_samples, n_components) Logarithm of the
        posterior probabilities (or responsibilities) of the point of each
        sample in X.
    """
    _estimate_gaussian_parameters(X, tf.math.exp(log_resp), self.reg_covar,
                                  self.weights_, self.means_, self.covariances_)
    _compute_precision_cholesky(self.covariances_, self.precisions_cholesky_)

  def _estimate_log_prob(self, X):
    return _estimate_log_gaussian_prob(X, self.means_,
                                       self.precisions_cholesky_)

  def _estimate_log_weights(self):
    return tf.math.log(self.weights_)

  def _compute_lower_bound(self, _, log_prob_norm):
    return log_prob_norm

  def _n_parameters(self):
    """Return the number of free parameters in the model."""
    _, n_features = self.means_.shape
    cov_params = self.n_components * n_features * (n_features + 1) / 2.
    mean_params = n_features * self.n_components
    return int(cov_params + mean_params + self.n_components - 1)

  def predict(self, X):
    """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
    """
    X = _check_X(X, None, self.means_.shape[1])
    return tf.math.argmax(self._estimate_weighted_log_prob(X), axis=1)

  def predict_proba(self, X):
    """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
    """
    X = _check_X(X, None, self.means_.shape[1])
    _, log_resp = self._estimate_log_prob_resp(X)
    return tf.math.exp(log_resp)

  def _estimate_weighted_log_prob(self, X):
    """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Args: ----------
        X : array-like of shape (n_samples, n_features)

        Returns:
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
    """
    return self._estimate_log_prob(X) + self._estimate_log_weights()

  def _estimate_log_prob_resp(self, X):
    """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Args: ----------
        X : array-like of shape (n_samples, n_features)

        Returns:
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
    """
    weighted_log_prob = self._estimate_weighted_log_prob(X)
    log_prob_norm = tf.math.reduce_logsumexp(weighted_log_prob, axis=1)
    log_resp = weighted_log_prob - log_prob_norm[:, tf.newaxis]
    return log_prob_norm, log_resp

  def _print_verbose_msg_init_beg(self, n_init):
    """Print verbose message on initialization."""
    if self.verbose == 1:
      logging.info("Initialization %d", n_init)
    elif self.verbose >= 2:
      logging.info("Initialization %d", n_init)
      self._init_prev_time = time.time()
      self._iter_prev_time = self._init_prev_time

  def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
    """Print verbose message on initialization."""
    if n_iter % self.verbose_interval == 0:
      if self.verbose == 1:
        logging.info("  Iteration %d", n_iter)
      elif self.verbose >= 2:
        cur_time = time.time()
        logging.info("  Iteration %d\t time lapse %.5fs\t ll change %.5f",
                     n_iter, cur_time - self._iter_prev_time, diff_ll)
        self._iter_prev_time = cur_time

  def _print_verbose_msg_init_end(self, ll):
    """Print verbose message on the end of iteration."""
    if self.verbose == 1:
      logging.info("Initialization converged: %s", self.converged_)
    elif self.verbose >= 2:
      logging.info("Initialization converged: %s\t time lapse %.5fs\t ll %.5f",
                   self.converged_,
                   time.time() - self._init_prev_time, ll)
