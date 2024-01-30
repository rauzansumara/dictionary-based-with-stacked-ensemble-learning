# Import Package
import numpy as np
import pandas as pd 
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from numba import njit, prange
from numba.typed import List
from warnings import warn
from sktime.transformations.panel.reduce import Tabularizer
from pyts.bag_of_words import WordExtractor
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, GRU, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D, AveragePooling1D, TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import optimizers


class UnivariateTransformerMixin:

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : array
            Transformed array.

        """  # noqa: E501
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


@njit()
def _uniform_bins(sample_min, sample_max, n_samples, n_bins):
    bin_edges = np.empty((n_bins - 1, n_samples))
    for i in prange(n_samples):
        bin_edges[:, i] = np.linspace(
            sample_min[i], sample_max[i], n_bins + 1)[1:-1]
    return bin_edges


@njit()
def _digitize_1d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins, X[i], side='left')
    return X_digit


@njit()
def _digitize_2d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins[i], X[i], side='left')
    return X_digit


def _digitize(X, bins):
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    return X_binned.astype('int64')


@njit
def _reshape_with_nan(X, n_samples, lengths, max_length):
    X_fill = np.full((n_samples, max_length), np.nan)
    for i in prange(n_samples):
        X_fill[i, :lengths[i]] = X[i]
    return X_fill


class KBinsDiscretizer(BaseEstimator, UnivariateTransformerMixin):

    def __init__(self, n_bins=5, strategy='quantile', raise_warning=True):
        self.n_bins = n_bins
        self.strategy = strategy
        self.raise_warning = raise_warning

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Bin the data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Binned data.

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        self._check_params(n_timestamps)

        bin_edges = self._compute_bins(
            X, n_samples, self.n_bins, self.strategy)
        X_new = _digitize(X, bin_edges)
        return X_new

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 (got {0})."
                .format(self.n_bins)
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0}).".format(self.strategy))

    def _compute_bins(self, X, n_samples, n_bins, strategy):
        if strategy == 'normal':
            bin_edges = norm.ppf(np.linspace(0, 1, self.n_bins + 1)[1:-1])
        elif strategy == 'uniform':
            sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
            bin_edges = _uniform_bins(
                sample_min, sample_max, n_samples, n_bins).T
        else:
            bin_edges = np.percentile(
                X, np.linspace(0, 100, self.n_bins + 1)[1:-1], axis=1
            ).T
            mask = np.c_[
                ~np.isclose(0, np.diff(bin_edges, axis=1), rtol=0, atol=1e-8),
                np.full((n_samples, 1), True)
            ]
            if (self.n_bins > 2) and np.any(~mask):
                samples = np.where(np.any(~mask, axis=1))[0]
                if self.raise_warning:
                    warn("Some quantiles are equal. The number of bins will "
                         "be smaller for sample {0}. Consider decreasing the "
                         "number of bins or removing these samples."
                         .format(samples), UserWarning)
                lengths = np.sum(mask, axis=1)
                max_length = np.max(lengths)

                bin_edges_ = List()
                for i in range(n_samples):
                    bin_edges_.append(bin_edges[i][mask[i]])

                bin_edges = _reshape_with_nan(bin_edges_, n_samples,
                                              lengths, max_length)
        return bin_edges


class SymbolicAggregateApproximation(BaseEstimator,
                                     UnivariateTransformerMixin):

    def __init__(self, n_bins=4, strategy='quantile', raise_warning=True,
                 alphabet=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.raise_warning = raise_warning
        self.alphabet = alphabet

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored
        y
            Ignored

        """
        return self

    def transform(self, X):
        """Bin the data with the given alphabet.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array, shape = (n_samples, n_timestamps)
            Binned data.

        """
        X = check_array(X, dtype='float64')
        n_timestamps = X.shape[1]
        alphabet = self._check_params(n_timestamps)
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, strategy=self.strategy,
            raise_warning=self.raise_warning
        )
        indices = discretizer.fit_transform(X)
        if isinstance(alphabet, str):
            return indices
        else:
            return alphabet[indices]

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins <= 26:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 and lower than "
                "or equal to 26 (got {0})."
                .format(self.n_bins)
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strategy))
        if not ((self.alphabet is None)
                or (self.alphabet == 'ordinal')
                or (isinstance(self.alphabet, (list, tuple, np.ndarray)))):
            raise TypeError("'alphabet' must be None, 'ordinal' or array-like "
                            "with shape (n_bins,) (got {0})"
                            .format(self.alphabet))
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])
        elif self.alphabet == 'ordinal':
            alphabet = 'ordinal'
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.n_bins,):
                raise ValueError("If 'alphabet' is array-like, its shape "
                                 "must be equal to (n_bins,).")
        return alphabet


def Symbolic2Words(X_train, y_train, X_test, y_test, n_bins, window_size):

    # SAX transformation
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
    X_sax_train = sax.fit_transform(Tabularizer().fit_transform(X_train))
    X_sax_test = sax.fit_transform(Tabularizer().fit_transform(X_test))

    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(pd.factorize(y_train)[0])
    test_labels_one_hot = to_categorical(pd.factorize(y_test)[0])

    # Bag-of-words transformation
    word = WordExtractor(window_size=window_size, numerosity_reduction=False)
    X_bow_train = word.transform(X_sax_train)
    X_bow_test = word.transform(X_sax_test)

    # tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_bow_train.tolist() + X_bow_test.tolist()) 

    max_length = 700
    n_output = train_labels_one_hot.shape[1]

    textvector = TextVectorization(max_tokens = len(tokenizer.word_index),
                                output_mode = 'int', output_sequence_length = max_length)
    textvector.adapt(X_bow_train.tolist() + X_bow_test.tolist())  

    # length of tokenizer
    input_dim = len(tokenizer.word_index) + 1

    return [textvector, input_dim, max_length, n_output, 
           X_bow_train, train_labels_one_hot, X_bow_test, test_labels_one_hot]


# define model
def SGCNN(textvector, input_dim, max_length, n_output):

    inputs = Input(shape=(1,), dtype = 'string')
    x = textvector(inputs)
    x = Embedding(input_dim, output_dim=4, input_length = max_length)(x)
    x = GRU(16, return_sequences=True)(x)
    x = Conv1D(16, 1, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32)(x)
    x = Dropout(0.1)(x)
    x = Dense(n_output, activation="softmax")(x)

    model = Model(inputs = inputs, outputs = x)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics='accuracy')
    
    return model