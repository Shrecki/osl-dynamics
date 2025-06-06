"""Base class for handling data.

"""

import re
import logging
import os
import pathlib
import pickle
import random
from contextlib import contextmanager
from shutil import rmtree
from dataclasses import dataclass

import numpy as np
from pqdm.threads import pqdm
from tqdm.auto import tqdm

from osl_dynamics.data import processing, rw, tf as dtf
from osl_dynamics.utils import misc

import tensorflow_probability as tfp
from osl_dynamics.inference import batched_cov


_logger = logging.getLogger("osl-dynamics")


class Data:
    """Data Class.

    The Data class enables the input and processing of data. When given a list
    of files, it produces a set of numpy memory maps which contain their raw
    data. It also provides methods for batching data and creating TensorFlow
    Datasets.

    Parameters
    ----------
    inputs : list of str or str or np.ndarray
        - A path to a directory containing :code:`.npy` files. Each
          :code:`.npy` file should be a subject or session.
        - A list of paths to :code:`.npy`, :code:`.mat` or :code:`.fif` files.
          Each file should be a subject or session. If a :code:`.fif` file is
          passed is must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
        - A numpy array. The array will be treated as continuous data from the
          same subject.
        - A list of numpy arrays. Each numpy array should be the data for a
          subject or session.

        The data files or numpy arrays should be in the format (n_samples,
        n_channels). If your data is in (n_channels, n_samples) format, use
        :code:`time_axis_first=False`.
    data_field : str, optional
        If a MATLAB (:code:`.mat`) file is passed, this is the field that
        corresponds to the time series data. By default we read the field
        :code:`'X'`. If a numpy (:code:`.npy`) or fif (:code:`.fif`) file is
        passed, this is ignored.
    picks : str or list of str, optional
        Only used if a fif file is passed. We load the data using the
        `mne.io.Raw.get_data <https://mne.tools/stable/generated/mne.io\
        .Raw.html#mne.io.Raw.get_data>`_ method. We pass this argument to the
        :code:`Raw.get_data` method. By default :code:`picks=None` retrieves
        all channel types.
    reject_by_annotation : str, optional
        Only used if a fif file is passed. We load the data using the
        `mne.io.Raw.get_data <https://mne.tools/stable/generated/mne.io\
        .Raw.html#mne.io.Raw.get_data>`_ method. We pass this argument to the
        :code:`Raw.get_data` method. By default
        :code:`reject_by_annotation=None` retrieves all time points. Use
        :code:`reject_by_annotation="omit"` to remove segments marked as bad.
    sampling_frequency : float, optional
        Sampling frequency of the data in Hz.
    mask_file : str, optional
        Path to mask file used to source reconstruct the data.
    parcellation_file : str, optional
        Path to parcellation file used to source reconstruct the data.
    time_axis_first : bool, optional
        Is the input data of shape (n_samples, n_channels)? Default is
        :code:`True`. If your data is in format (n_channels, n_samples), use
        :code:`time_axis_first=False`.
    load_memmaps : bool, optional
        Should we load the data as memory maps (memmaps)? If :code:`True`, we
        will load store the data on disk rather than loading it into memory.
    store_dir : str, optional
        If `load_memmaps=True`, then we save data to disk and load it as
        a memory map. This is the directory to save the memory maps to.
        Default is :code:`./tmp`.
    buffer_size : int, optional
        Buffer size for shuffling a TensorFlow Dataset. Smaller values will lead
        to less random shuffling but will be quicker. Default is 100000.
    use_tfrecord : bool, optional
        Should we save the data as a TensorFlow Record? This is recommended for
        training on large datasets. Default is :code:`False`.
    session_labels : list of SessionLabels, optional
        Extra session labels.
    n_jobs : int, optional
        Number of processes to load the data in parallel.
        Default is 1, which loads data in serial.
    """

    def __init__(
        self,
        inputs,
        data_field="X",
        picks=None,
        reject_by_annotation=None,
        sampling_frequency=None,
        mask_file=None,
        parcellation_file=None,
        time_axis_first=True,
        load_memmaps=False,
        store_dir="tmp",
        buffer_size=4000,
        use_tfrecord=False,
        session_labels=None,
        n_jobs=1,
    ):
        self._identifier = id(self)
        self.data_field = data_field
        self.picks = picks
        self.reject_by_annotation = reject_by_annotation
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.mask_file = mask_file
        self.parcellation_file = parcellation_file
        self.time_axis_first = time_axis_first
        self.load_memmaps = load_memmaps
        self.buffer_size = buffer_size
        self.use_tfrecord = use_tfrecord
        self.n_jobs = n_jobs

        # Validate inputs
        self.inputs = rw.validate_inputs(inputs)

        if len(self.inputs) == 0:
            raise ValueError("No valid inputs were passed.")

        # Directory to store memory maps created by this class
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        # Load and validate the raw data
        self.raw_data_arrays, self.raw_data_filenames = self.load_raw_data()
        self.validate_data()

        self.n_raw_data_channels = self.raw_data_arrays[0].shape[-1]

        # Get data preparation attributes if there's a pickle file in the
        # input directory
        if not isinstance(inputs, list):
            self.load_preparation(inputs)

        # Store raw data in the arrays attribute
        self.arrays = self.raw_data_arrays

        # Create filenames for prepared data memmaps
        prepared_data_pattern = "prepared_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(self.n_sessions)), identifier=self._identifier
        )
        self.prepared_data_filenames = [
            str(self.store_dir / prepared_data_pattern.format(i=i))
            for i in range(self.n_sessions)
        ]

        # Arrays to keep when making TensorFlow Datasets
        self.keep = list(range(self.n_sessions))

        # Extra session labels
        if session_labels is None:
            self.session_labels = []
            
        self.n_orig_channels = None  # Will store the original value

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, item):
        return self.arrays[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_sessions: {self.n_sessions}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self):
        """Return raw data as a list of arrays."""
        return self.raw_data_arrays

    @property
    def n_channels(self):
        """Number of channels in the data files."""
        if self.n_orig_channels is not None:
            return self.n_orig_channels
        else:
            return self.arrays[0].shape[-1]

    @property
    def n_samples(self):
        """Number of samples across all arrays."""
        return sum([array.shape[-2] for array in self.arrays])

    @property
    def n_sessions(self):
        """Number of arrays."""
        return len(self.arrays)

    @contextmanager
    def set_keep(self, keep):
        """Context manager to temporarily set the kept arrays.

        Parameters
        ----------
        keep : int or list of int
            Indices to keep in the Data.arrays list.
        """
        # Store the current kept arrays
        current_keep = self.keep
        try:
            if isinstance(keep, int):
                keep = [keep]
            if not isinstance(keep, list):
                raise ValueError("keep must be a list of indices or a single index.")

            # Set the new kept arrays
            self.keep = keep
            yield
        finally:
            self.keep = current_keep

    def set_sampling_frequency(self, sampling_frequency):
        """Sets the :code:`sampling_frequency` attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency

    def set_buffer_size(self, buffer_size):
        """Set the :code:`buffer_size` attribute.

        Parameters
        ----------
        buffer_size : int
            Buffer size for shuffling a TensorFlow Dataset. Smaller values will
            lead to less random shuffling but will be quicker.
        """
        self.buffer_size = buffer_size

    def time_series(self, prepared=True, concatenate=False):
        """Time series data for all arrays.

        Parameters
        ----------
        prepared : bool, optional
            Should we return the latest data after we have prepared it or
            the original data we loaded into the Data object?
        concatenate : bool, optional
            Should we return the time series for each array concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each array.
        """
        # What data should we return?
        if prepared:
            arrays = self.arrays
        else:
            arrays = self.raw_data_arrays

        # Should we return one long time series?
        if concatenate or self.n_sessions == 1:
            return np.concatenate(arrays)
        else:
            return arrays

    def load_raw_data(self):
        """Import data into a list of memory maps.

        Returns
        -------
        memmaps : list of np.memmap
            List of memory maps.
        raw_data_filenames : list of str
            List of paths to the raw data memmaps.
        """
        raw_data_pattern = "raw_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )
        raw_data_filenames = [
            str(self.store_dir / raw_data_pattern.format(i=i))
            for i in range(len(self.inputs))
        ]
        # self.raw_data_filenames is not used if self.inputs is a list of
        # strings, where the strings are paths to .npy files

        # Function to save a single memory map
        def _make_memmap(raw_data, mmap_location):
            if not self.load_memmaps:  # do not load into the memory maps
                mmap_location = None
            raw_data_mmap = rw.load_data(
                raw_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
                mmap_location,
                mmap_mode="r",
            )
            if not self.time_axis_first:
                raw_data_mmap = raw_data_mmap.T
            return raw_data_mmap

        # Load data
        memmaps = pqdm(
            array=zip(self.inputs, raw_data_filenames),
            function=_make_memmap,
            n_jobs=self.n_jobs,
            desc="Loading files",
            argument_type="args",
            total=len(self.inputs),
        )

        return memmaps, raw_data_filenames

    def validate_data(self):
        """Validate data files."""
        n_channels = [array.shape[-1] for array in self.raw_data_arrays]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def _validate_batching(
        self,
        sequence_length,
        batch_size,
        step_size=None,
        drop_last_batch=False,
        concatenate=True,
    ):
        """Validate the batching process.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
            Defaults to False.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
            Defaults to True.
        """

        # Calculate number of sequences per session
        n_sequences_per_session = [
            (array.shape[0] - sequence_length) // step_size + 1 for array in self.arrays
        ]

        # Calculate number of batches
        if concatenate:
            # Calculate total batches across concatenated sequences
            total_n_sequences = sum(n_sequences_per_session)
            n_batches = total_n_sequences // batch_size
            # Add one more batch if the last incomplete batch is not dropped
            if not drop_last_batch and total_n_sequences & batch_size != 0:
                n_batches += 1
        else:
            # Calculate batches per session individually, then sum
            n_batches_per_session = [
                n // batch_size + (0 if drop_last_batch or n % batch_size == 0 else 1)
                for n in n_sequences_per_session
            ]
            n_batches = sum(n_batches_per_session)

        if n_batches < 1:
            raise ValueError(
                "Number of batches must be greater than or equal to 1. "
                + "Please adjust your sequence length and batch size."
            )

    def select(self, channels=None, sessions=None, use_raw=False):
        """Select channels.

        This is an in-place operation.

        Parameters
        ----------
        channels : int or list of int, optional
            Channel indices to keep. If None, all channels are retained.
        sessions : int or list of int, optional
            Session indices to keep. If None, all sessions are retained.
        use_raw : bool, optional
            Should we select channel from the original 'raw' data that
            we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if channels is None:
            # Keep all channels
            if use_raw:
                n_channels = self.raw_data_arrays[0].shape[-1]
            else:
                n_channels = self.arrays[0].shape[-1]
            channels = range(n_channels)

        if sessions is None:
            # Keep all sessions
            if use_raw:
                n_sessions = len(self.raw_data_arrays)
            else:
                n_sessions = len(self.arrays)
            sessions = range(n_sessions)

        if isinstance(channels, int):
            channels = [channels]

        if isinstance(sessions, int):
            sessions = [sessions]

        if isinstance(channels, range):
            channels = list(channels)

        if isinstance(sessions, range):
            sessions = list(sessions)

        if not isinstance(channels, list):
            raise ValueError("channels must be an int or list of int.")

        if not isinstance(sessions, list):
            raise ValueError("sessions must be an int or list of int.")

        # What data should we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Select channels
        new_arrays = []
        for i in tqdm(sessions, desc="Selecting channels/sessions"):
            array = arrays[i][:, channels]
            if self.load_memmaps:
                array = misc.array_to_memmap(self.prepared_data_filenames[i], array)
            new_arrays.append(array)
        self.arrays = new_arrays

        return self

    def filter(self, low_freq=None, high_freq=None, use_raw=False):
        """Filter the data.

        This is an in-place operation.

        Parameters
        ----------
        low_freq : float, optional
            Frequency in Hz for a high pass filter. If :code:`None`, no high
            pass filtering is applied.
        high_freq : float, optional
            Frequency in Hz for a low pass filter. If :code:`None`, no low pass
            filtering is applied.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if low_freq is None and high_freq is None:
            _logger.warning("No filtering applied.")
            return

        if self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the "
                "data. Use Data.set_sampling_frequency() or pass "
                "Data(..., sampling_frequency=...) when creating the Data "
                "object."
            )

        self.low_freq = low_freq
        self.high_freq = high_freq

        # Function to apply filtering to a single array
        def _apply(array, prepared_data_file):
            array = processing.temporal_filter(
                array, low_freq, high_freq, self.sampling_frequency
            )
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Filtering",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def downsample(self, freq, use_raw=False):
        """Downsample the data.

        This is an in-place operation.

        Parameters
        ----------
        freq : float
            Frequency in Hz to downsample to.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the "
                "data. Use Data.set_sampling_frequency() or pass "
                "Data(..., sampling_frequency=...) when creating the Data "
                "object."
            )

        if use_raw:
            sampling_frequency = self.original_sampling_frequency
        else:
            sampling_frequency = self.sampling_frequency

        # Function to apply downsampling to a single array
        def _apply(array, prepared_data_file):
            array = processing.downsample(array, freq, sampling_frequency)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Downsampling",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        # Update sampling_frequency attributes
        self.sampling_frequency = freq

        return self

    def pca(
        self,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
        use_raw=False,
        component_offset=0
    ):
        """Principal component analysis (PCA).

        This function will first standardize the data then perform PCA.
        This is an in-place operation.

        Parameters
        ----------
        n_pca_components : int, optional
            Number of PCA components to keep. If :code:`None`, then
            :code:`pca_components` should be passed.
        pca_components : np.ndarray, optional
            PCA components to apply if they have already been calculated.
            If :code:`None`, then :code:`n_pca_components` should be passed.
        whiten : bool, optional
            Should we whiten the PCA'ed data?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        component_offset: int, optional
            From which offset should we include components

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if (n_pca_components is None and pca_components is None) or (
            n_pca_components is not None and pca_components is not None
        ):
            raise ValueError("Please pass either n_pca_components or pca_components.")
        
        if component_offset < 0:
            raise ValueError("Please pass a non-negative component_offset.")
        if not isinstance(component_offset, int):
            raise ValueError("component_offset must be a non-negative integer.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise ValueError("pca_components must be a numpy array.")

        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.component_offset = component_offset
        self.whiten = whiten

        # What data should we apply PCA to?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate PCA
        if n_pca_components is not None:
            # Calculate covariance of the data
            n_channels = arrays[0].shape[-1]
            covariance = np.zeros([n_channels, n_channels])
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                covariance += np.transpose(std_data) @ std_data

            # Use SVD on the covariance to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, component_offset:component_offset+n_pca_components].astype(np.float32)
            self.explained_variance = np.sum(s[component_offset:component_offset+n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s[component_offset:component_offset+n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        # Function to apply PCA to a single array
        def _apply(array, prepared_data_file):
            array = processing.standardize(array)
            array = array @ self.pca_components
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply PCA in parallel
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="PCA",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def tde(self, n_embeddings, use_raw=False):
        """Time-delay embedding (TDE).

        This is an in-place operation.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings

        # Function to apply TDE to a single array
        def _apply(array, prepared_data_file):
            array = processing.time_embed(array, n_embeddings)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply TDE in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="TDE",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def tde_pca(
        self,
        n_embeddings,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
        use_raw=False,
        component_offset=0
    ):
        """Time-delay embedding (TDE) and principal component analysis (PCA).

        This function will first standardize the data, then perform TDE then
        PCA. It is useful to do both operations in a single methods because
        it avoids having to save the time-embedded data. This is an in-place
        operation.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        n_pca_components : int, optional
            Number of PCA components to keep. If :code:`None`, then
            :code:`pca_components` should be passed.
        pca_components : np.ndarray, optional
            PCA components to apply if they have already been calculated.
            If :code:`None`, then :code:`n_pca_components` should be passed.
        whiten : bool, optional
            Should we whiten the PCA'ed data?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        component_offset: int, optional
            From which offset should we include components

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if (n_pca_components is None and pca_components is None) or (
            n_pca_components is not None and pca_components is not None
        ):
            raise ValueError("Please pass either n_pca_components or pca_components.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise ValueError("pca_components must be a numpy array.")
        
        
        if component_offset < 0:
            raise ValueError("Please pass a non-negative component_offset.")
        if not isinstance(component_offset, int):
            raise ValueError("component_offset must be a non-negative integer.")


        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten
        self.component_offset = component_offset

        # What data should we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays
        self.n_te_channels = arrays[0].shape[-1] * n_embeddings

        # Calculate PCA on TDE data
        if n_pca_components is not None:
            # Calculate covariance of the data
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                te_std_data = processing.time_embed(std_data, n_embeddings)
                covariance += np.transpose(te_std_data) @ te_std_data

            # Use SVD on the covariance to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, component_offset:component_offset+n_pca_components].astype(np.float32)
            self.explained_variance = np.sum(s[component_offset:component_offset+n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s[component_offset:component_offset+n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        # Function to apply TDE-PCA to a single array
        def _apply(array, prepared_data_file):
            array = processing.standardize(array)
            array = processing.time_embed(array, n_embeddings)
            array = array @ self.pca_components
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply TDE and PCA in parallel
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="TDE-PCA",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def amplitude_envelope(self, use_raw=False):
        """Calculate the amplitude envelope.

        This is an in-place operation.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        """

        # Function to calculate amplitude envelope for a single array
        def _apply(array, prepared_data_file):
            array = processing.amplitude_envelope(array)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Amplitude envelope",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def moving_average(self, n_window, use_raw=False):
        """Calculate a moving average.

        This is an in-place operation.

        Parameters
        ----------
        n_window : int
            Number of data points in the sliding window. Must be odd.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.n_window = n_window

        # Function to apply sliding window to a single array
        def _apply(array, prepared_data_file):
            array = processing.moving_average(array, n_window)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Sliding window",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def standardize(self, use_raw=False):
        """Standardize (z-transform) the data.

        This is an in-place operation.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        """

        # Function to apply standardisation to a single array
        def _apply(array):
            return processing.standardize(array, create_copy=False)

        # Apply standardisation to each array in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        self.arrays = pqdm(
            array=zip(arrays),
            function=_apply,
            desc="Standardize",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self
    
    def moving_covar_cholesky_vectorized(self,n_window,use_raw=False, approach="naive", batch_size=None):
        """Sliding-window covariance.
        
        This function will compute a sliding-window covariance, per array,
        using n_window samples for each covariance matrix.
        Covariance matrices are then decomposed as their vectorized Cholesky
        factors, to reduce storage requirements.
        
        This is an in-place operation.

        Parameters
        ----------
        n_window : int
            Number of samples to compute any covariance matrix. Must be at least equal to number of channels.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        approach: string, optional
            Approach to use, can be naive, fft, batch_fft, batch_cython
        batch_size: int, optional
            Batch size. Only used in batch approaches.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if n_window is None:
            raise ValueError("Please pass n_window.")
        if not isinstance(n_window, int) or n_window < 1:
            raise ValueError("n_window should be a non-negative integer.")
        
        if n_window < self.n_channels:
            raise ValueError("n_window should be at least the number of channels for covariances to be defined.")
        
        self.n_window = n_window
        
        self.n_orig_channels = self.n_channels

        # What data should we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays
        
        
        def fft_moving_average(array, L):
            """
            Compute the moving average of a 1D array x with window length L using FFT convolution.
            """
            n = len(array)
            kernel = np.ones(L) / L
            # Next power of 2 for zero-padding (efficient FFT computation)
            nfft = np.power(2, np.ceil(np.log2(n + L - 1)).astype(int))
            array_fft = np.fft.rfft(array, n=nfft)
            kernel_fft = np.fft.rfft(kernel, n=nfft)
            conv = np.fft.irfft(array_fft * kernel_fft, n=nfft)[:n + L - 1]
            # We only need the 'valid' part: indices L-1 to n-1
            return conv[L-1:n]

        def sliding_cov_fft(array, L):
            """
            Compute sliding-window covariance matrices using FFT-based convolution.
            X: np.ndarray of shape (n_samples, n_vars)
            L: window length
            
            Returns:
                covs: list of covariance matrices for each window (length n_samples - L + 1)
            """
            n_samples, n_vars = array.shape
            valid_length = n_samples - L + 1
            
            # Compute moving averages for each variable
            means = np.empty((valid_length, n_vars))
            for j in range(n_vars):
                means[:, j] = fft_moving_average(array[:, j], L)
            
            # Precompute moving averages for products: for each pair (i,j)
            # We'll store the covariance matrices for each window
            covs = np.zeros((n_vars, n_vars,valid_length))
            
            # Compute for diagonal elements first (variance)
            for j in range(n_vars):
                moving_prod = fft_moving_average(array[:, j] * array[:, j], L)
                # Covariance for variable j with itself: E[x^2] - (E[x])^2
                covs_diag = moving_prod - means[:, j]**2
                covs[j,j,:] = covs_diag

                
            # Now compute off-diagonals (symmetric)
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    moving_prod = fft_moving_average(array[:, i] * array[:, j], L)
                    cov_ij = moving_prod - means[:, i] * means[:, j]
                    covs[i,j] = cov_ij
                    covs[j,i] = cov_ij
            covs *= L
            covs /= (L-1)
            return covs    

        def cholesky_vectorize(cov):
            import tensorflow_probability as tfp
            cov = np.transpose(cov, (2, 0, 1))
            chol_np = tfp.math.fill_triangular_inverse(np.linalg.cholesky(cov)).numpy()
            return chol_np


        def compute_sliding_covariances_batches(array, batch_size):
            """
            Compute sliding-window covariance matrices in batches and return
            their vectorized Cholesky factors. The computation is done in batches
            over the "valid" sliding-window outputs, so no edge effects are introduced.
            
            Parameters
            ----------
            array : np.ndarray
                Input data of shape (n_samples, n_vars).
            batch_size : int
                Number of windows to process in each batch.
            
            Yields
            ------
            chol_vectorized_batch : np.ndarray
                Batch of vectorized Cholesky factors, of shape 
                (batch_size_current, n_triangular_elements) where 
                n_triangular_elements = n_vars*(n_vars+1)//2.
            """

            n_samples, n_vars = array.shape
            valid_length = n_samples - n_window + 1  # number of valid sliding windows
            
            # Precompute moving averages for each variable over the full valid region.
            # For a given channel j, we compute the convolution with a boxcar kernel.
            kernel = np.ones(n_window) / n_window
            means = np.empty((valid_length, n_vars))
            for j in range(n_vars):
                # np.convolve(x, kernel, mode='valid') returns the moving average over all valid windows.
                means[:, j] = np.convolve(array[:, j], kernel, mode='valid')
            
            # Process valid windows in batches.
            for start in range(0, valid_length, batch_size):
                end = min(start + batch_size, valid_length)
                batch_length = end - start
                
                # Initialize an array to hold covariance matrices for this batch.
                # Shape: (n_vars, n_vars, batch_length)
                covs_batch = np.zeros((n_vars, n_vars, batch_length))
                
                # Diagonal entries: For each variable j, compute moving average of x^2.
                for j in range(n_vars):
                    moving_prod = np.convolve(array[:, j]**2, kernel, mode='valid')[start:end]
                    # Sample covariance formula: Var = (E[x^2] - (E[x])^2) scaled appropriately.
                    covs_batch[j, j, :] = moving_prod - means[start:end, j]**2

                # Off-diagonal entries: For each pair (i,j), i < j.
                for i in range(n_vars):
                    for j in range(i+1, n_vars):
                        moving_prod = np.convolve(array[:, i] * array[:, j], kernel, mode='valid')[start:end]
                        cov_ij = moving_prod - means[start:end, i] * means[start:end, j]
                        covs_batch[i, j, :] = cov_ij
                        covs_batch[j, i, :] = cov_ij
                
                # Scale to get the unbiased sample covariance.
                # Note: Using the fact that 
                #   unbiased_cov = (L / (L-1)) * (E[x^2] - (E[x])^2)
                covs_batch = covs_batch * n_window / (n_window - 1)
                
                # Transpose to have shape (batch_length, n_vars, n_vars)
                covs_batch = np.transpose(covs_batch, (2, 0, 1))
                
                # Compute the Cholesky decomposition for each covariance matrix in the batch.
                # Here we assume the covariances are positive-definite.
                chol_batch = np.linalg.cholesky(covs_batch)
                
                # Vectorize the lower-triangular Cholesky factors using TensorFlow Probability.
                # tfp.math.fill_triangular_inverse orders the entries row-by-row.
                chol_vectorized_batch = tfp.math.fill_triangular_inverse(chol_batch).numpy()
                
                yield chol_vectorized_batch

        # Function to compute covariance
        def _apply(array, prepared_data_file):
            import tensorflow_probability as tfp
            if approach == "fft":
                # Compute with FFT covariances
                cov = sliding_cov_fft(array, self.n_window)
                
                # Decompose each covariance as vectorized Cholesky factor
                array = cholesky_vectorize(cov)
            elif approach == "batch_fft": 
                all_batches = []
                i = 0
                print(f"Batch size: {batch_size}")
                for batch in compute_sliding_covariances_batches(array,batch_size):
                    print(f"batch {i} done")
                    i = i+1
                    all_batches.append(batch)
                array = np.concatenate(all_batches, axis=0)               
            elif approach == "naive":
                n_samples, n_vars = array.shape
                valid_length = n_samples - n_window + 1
                cholesky_res = np.zeros((valid_length, int(n_vars*(n_vars+1)/2)))
                for i in tqdm(range(valid_length), desc="Timestep"):
                    cholesky_res[i] = tfp.math.fill_triangular_inverse(np.linalg.cholesky(np.cov(array[i:i+n_window], rowvar=False)))
                array = cholesky_res
            elif approach =="batch_cython":
                array = batched_cov.batched_covariance_and_cholesky(array.astype(np.float64),int(n_window), int(batch_size))[1]
                n_samples,n_vars = array.shape
                # Reorder from "numpy-style" vectorization order to tensorflow-style vectorization order
                def get_tril_to_tfp_indices(n_channels):
                    """
                    Create a direct mapping of indices from np.tril_indices ordering to 
                    TensorFlow Probability's FillTriangular ordering.
                    
                    Parameters
                    ----------
                    n_channels : int
                        Number of channels/dimension of the square matrix.
                    
                    Returns
                    -------
                    numpy.ndarray
                        Array of indices that can be used to reorder vectors from 
                        np.tril_indices ordering to TFP's FillTriangular ordering.
                    """
                    import tensorflow as tf
                    vec_size = n_channels * (n_channels + 1) // 2
                    #print(test_seq)
                    # Create TFP ordering
                    fill_triangular = tfp.bijectors.FillTriangular()
                    test_vector = np.arange(vec_size, dtype=int)
                    test_matrix = fill_triangular(tf.convert_to_tensor([test_vector])).numpy()[0]
                    
                    #print(test_matrix)
                    
                    ids_remap = np.zeros(vec_size,dtype=int)
                    for j in range(n_channels):
                        for k in range(j+1):
                            ids_remap[j*(j+1)//2 + k] = test_matrix[j, k]
                    #print(ids_remap)
                    
                    return np.argsort(ids_remap)
                def reorder_tril_to_tfp_direct(cholesky_vectors, transpose_needed=False, n_channels=None, mapping=None):
                    """
                    Reorder vectorized Cholesky factors from np.tril_indices ordering 
                    to TensorFlow Probability's FillTriangular ordering using direct indexing.
                    
                    Parameters
                    ----------
                    cholesky_vectors : numpy.ndarray
                        Vectorized Cholesky factors using np.tril_indices ordering.
                        Shape is (vec_size, T) where vec_size = n_channels*(n_channels+1)/2
                        and T is the number of time points. Can also accept shape (T, vec_size).
                    
                    n_channels : int, optional
                        Number of channels/dimension of the square matrix. 
                        Required if mapping is not provided.
                    
                    mapping : numpy.ndarray, optional
                        Precomputed index mapping from get_tril_to_tfp_indices(). 
                        If not provided, it will be computed using n_channels.
                    
                    Returns
                    -------
                    numpy.ndarray
                        Reordered vectorized Cholesky factors in TFP's FillTriangular ordering.
                        Shape matches the input shape.
                    """
                    # Check if we need to transpose
                    if cholesky_vectors.ndim == 2 and transpose_needed:
                        cholesky_vectors = cholesky_vectors.T
                        
                    # Get mapping if not provided
                    if mapping is None:
                        if n_channels is None:
                            raise ValueError("Either mapping or n_channels must be provided")
                        mapping = get_tril_to_tfp_indices(n_channels)
                    
                    # Apply the mapping to reorder the vectors
                    if cholesky_vectors.ndim == 1:
                        # Single vector
                        reordered = cholesky_vectors[mapping]
                    else:
                        # Multiple vectors
                        reordered = cholesky_vectors[mapping, :]
                    
                    # Transpose back if needed
                    if transpose_needed:
                        reordered = reordered.T
                    
                    return reordered
                mapping = get_tril_to_tfp_indices(self.n_channels)
                array = reorder_tril_to_tfp_direct(array,transpose_needed=True,n_channels=self.n_channels, mapping=mapping)
                
            else:
                raise NotImplementedError(f"approach can only be fft, batch_fft or naive, but was {approach}")
            # Return result
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array
        
        for i in tqdm(range(len(arrays)), desc="Sliding window covariance"):
            array = arrays[i]
            self.arrays[i] = _apply(array,self.prepared_data_filenames)
            #print(f"N channels after apply: {self.n_channels}")

        self.n_covar_channels = self.arrays[0].shape[1]
        
        
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def align_channel_signs(
        self,
        template_data=None,
        template_cov=None,
        n_init=3,
        n_iter=2500,
        max_flips=20,
        n_embeddings=1,
        standardize=True,
        use_raw=False,
    ):
        """Align the sign of each channel across sessions.

        If no template data/covariance is passed, we use the median session.

        Parameters
        ----------
        template_data : np.ndarray or str, optional
            Data to align the sign of channels to.
            If :code:`str`, the file will be read in the same way as the
            inputs to the Data object.
        template_cov : np.ndarray or str, optional
            Covariance to align the sign of channels. This must be the
            covariance of the time-delay embedded data.
            If :code:`str`, must be the path to a :code:`.npy` file.
        n_init : int, optional
            Number of initializations.
        n_iter : int, optional
            Number of sign flipping iterations per subject to perform.
        max_flips : int, optional
            Maximum number of channels to flip in an iteration.
        n_embeddings : int, optional
            We may want to compare the covariance of time-delay embedded data
            when aligning the signs. This is the number of embeddings. The
            returned data is not time-delay embedded.
        standardize : bool, optional
            Should we standardize the data before comparing across sessions?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if template_data is not None and template_cov is not None:
            raise ValueError(
                "Only pass one of the arguments template_data or template_cov, "
                "not both."
            )

        if self.n_channels < max_flips:
            _logger.warning(
                f"max_flips={max_flips} cannot be greater than "
                f"n_channels={self.n_channels}. "
                f"Setting max_flips={self.n_channels}."
            )
            max_flips = self.n_channels

        if isinstance(template_data, str):
            template_data = rw.load_data(
                template_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
                memmap_location=None,
                mmap_mode="r",
            )
            if not self.time_axis_first:
                template_data = template_data.T

        if isinstance(template_cov, str):
            template_cov = np.load(template_cov)

        # Helper functions
        def _calc_cov(array):
            array = processing.time_embed(array, n_embeddings)
            if standardize:
                array = processing.standardize(array, create_copy=False)
            return np.cov(array.T)

        def _calc_corr(M1, M2, mode=None):
            if mode == "abs":
                M1 = np.abs(M1)
                M2 = np.abs(M2)
            m, n = np.triu_indices(M1.shape[0], k=n_embeddings)
            M1 = M1[m, n]
            M2 = M2[m, n]
            return np.corrcoef([M1, M2])[0, 1]

        def _calc_metrics(covs):
            metric = np.zeros([self.n_sessions, self.n_sessions])
            for i in tqdm(range(self.n_sessions), desc="Comparing sessions"):
                for j in range(i + 1, self.n_sessions):
                    metric[i, j] = _calc_corr(covs[i], covs[j], mode="abs")
                    metric[j, i] = metric[i, j]
            return metric

        def _randomly_flip(flips, max_flips):
            n_channels_to_flip = np.random.choice(max_flips, size=1)
            random_channels_to_flip = np.random.choice(
                self.n_channels, size=n_channels_to_flip, replace=False
            )
            new_flips = np.copy(flips)
            new_flips[random_channels_to_flip] *= -1
            return new_flips

        def _apply_flips(cov, flips):
            flips = np.repeat(flips, n_embeddings)[np.newaxis, ...]
            flips = flips.T @ flips
            return cov * flips

        def _find_and_apply_flips(cov, tcov, array, ind):
            best_flips = np.ones(self.n_channels)
            best_metric = 0
            for n in range(n_init):
                flips = np.ones(self.n_channels)
                metric = _calc_corr(cov, tcov)
                for j in range(n_iter):
                    new_flips = _randomly_flip(flips, max_flips)
                    new_cov = _apply_flips(cov, new_flips)
                    new_metric = _calc_corr(new_cov, tcov)
                    if new_metric > metric:
                        flips = new_flips
                        metric = new_metric
                if metric > best_metric:
                    best_metric = metric
                    best_flips = flips
                _logger.info(
                    f"Session {ind}, Init {n}, best correlation with template: "
                    f"{best_metric:.3f}"
                )
            return array * best_flips[np.newaxis, ...].astype(np.float32)

        # What data do we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate covariance of each session
        covs = pqdm(
            array=zip(arrays),
            function=_calc_cov,
            desc="Calculating covariances",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in covs]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        # Calculate/get template covariances
        if template_cov is None:
            metrics = _calc_metrics(covs)
            metrics_sum = np.sum(metrics, axis=1)
            argmedian = np.argsort(metrics_sum)[len(metrics_sum) // 2]
            _logger.info(f"Using session {argmedian} as template")
            template_cov = covs[argmedian]

        if template_data is not None:
            template_cov = _calc_cov(template_data)

        # Perform the sign flipping
        _logger.info("Aligning channel signs across sessions")
        tcovs = [template_cov] * self.n_sessions
        indices = range(self.n_sessions)
        self.arrays = pqdm(
            array=zip(covs, tcovs, arrays, indices),
            function=_find_and_apply_flips,
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
            disable=True,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def remove_bad_segments(
        self,
        window_length=None,
        significance_level=0.05,
        maximum_fraction=0.1,
        use_raw=False,
    ):
        """Automated bad segment removal using the G-ESD algorithm.

        Parameters
        ----------
        window_length : int, optional
            Window length to used to calculate statistics.
            Defaults to twice the sampling frequency.
        significance_level : float, optional
            Significance level (p-value) to consider as an outlier.
        maximum_fraction : float, optional
            Maximum fraction of time series to mark as bad.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.gesd_window_length = window_length
        self.gesd_significance_level = significance_level
        self.gesd_maximum_fraction = maximum_fraction

        if window_length is None:
            if self.sampling_frequency is None:
                raise ValueError(
                    "window_length must be passed. Alternatively, set the "
                    "sampling frequency to use "
                    "window_length=2*sampling_frequency. The sampling "
                    "frequency can be set using Data.set_sampling_frequency() "
                    "or pass Data(..., sampling_frequency=...) when creating "
                    "the Data object."
                )
            else:
                window_length = 2 * self.sampling_frequency

        # Function to remove bad segments to a single array
        def _apply(array, prepared_data_file):
            array = processing.remove_bad_segments(
                array, window_length, significance_level, maximum_fraction
            )
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Run in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Bad segment removal",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def prepare(self, methods):
        """Prepare data.

        Wrapper for calling a series of data preparation methods. Any method
        in Data can be called. Note that if the same method is called multiple
        times, the method name should be appended with an underscore and a
        number, e.g. :code:`standardize_1` and :code:`standardize_2`.

        Parameters
        ----------
        methods : dict
            Each key is the name of a method to call. Each value is a
            :code:`dict` containing keyword arguments to pass to the method.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.

        Examples
        --------
        TDE-PCA data preparation::

            methods = {
                "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
                "standardize": {},
            }
            data.prepare(methods)

        Amplitude envelope data preparation::

            methods = {
                "filter": {"low_freq": 1, "high_freq": 45},
                "amplitude_envelope": {},
                "moving_average": {"n_window": 5},
                "standardize": {},
            }
            data.prepare(methods)
        """
        # Pattern for identifying the method name from "method-name_num"
        pattern = re.compile(r"^(\w+?)(_\d+)?$")

        for method_name, kwargs in methods.items():
            # Remove the "_num" part from the dict key
            method_name = pattern.search(method_name).groups()[0]

            # Apply method
            method = getattr(self, method_name)
            method(**kwargs)

        return self

    def trim_time_series(
        self,
        sequence_length=None,
        n_embeddings=None,
        n_window=None,
        prepared=True,
        concatenate=False,
        verbose=False,
    ):
        """Trims the data time series.

        Removes the data points that are lost when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int, optional
            Length of the segement of data to feed into the model.
            Can be pass to trim the time points that are lost when separating
            into sequences.
        n_embeddings : int, optional
            Number of data points used to embed the data. If :code:`None`,
            then we use :code:`Data.n_embeddings` (if it exists).
        n_window : int, optional
            Number of data points the sliding window applied to the data.
            If :code:`None`, then we use :code:`Data.n_window` (if it exists).
        prepared : bool, optional
            Should we return the prepared data? If not we return the raw data.
        concatenate : bool, optional
            Should we concatenate the data for each array?
        verbose : bool, optional
            Should we print the number of data points we're removing?

        Returns
        -------
        list of np.ndarray
            Trimed time series for each array.
        """
        # How many time points from the start/end of the time series should
        # we remove?
        n_remove = 0
        if n_embeddings is None:
            if hasattr(self, "n_embeddings"):
                n_remove += self.n_embeddings // 2
        else:
            n_remove += n_embeddings // 2
        if n_window is None:
            if hasattr(self, "n_window"):
                n_remove += self.n_window // 2
        else:
            n_remove += n_window // 2
        if verbose:
            _logger.info(
                f"Removing {n_remove} data points from the start and end"
                " of each array due to time embedding/sliding window."
            )

        # What data should we trim?
        if prepared:
            arrays = self.arrays
        else:
            arrays = self.raw_data_arrays

        trimmed_time_series = []
        for i, array in enumerate(arrays):
            # Remove data points lost to time embedding or sliding window
            if n_remove != 0:
                array = array[n_remove:-n_remove]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = array.shape[0] // sequence_length
                n_keep = n_sequences * sequence_length
                if verbose:
                    _logger.info(
                        f"Removing {array.shape[0] - n_keep} data points "
                        f"from the end of array {i} due to sequencing."
                    )
                array = array[:n_keep]

            trimmed_time_series.append(array)

        if concatenate or len(trimmed_time_series) == 1:
            trimmed_time_series = np.concatenate(trimmed_time_series)

        return trimmed_time_series

    def count_sequences(self, sequence_length, step_size=None):
        """Count sequences.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        step_size : int, optional
            The number of samples by which to move the sliding window between
            sequences. Defaults to :code:`sequence_length`.

        Returns
        -------
        n : np.ndarray
            Number of sequences for each session's data.
        """
        return np.array(
            [
                dtf.get_n_sequences(array, sequence_length, step_size)
                for array in self.arrays
            ]
        )

    def _create_data_dict(self, i, array):
        """Create a dictionary of data for a single session.

        Parameters
        ----------
        i : int
            Index of the session.
        array : np.ndarray
            Time series data for a single session.

        Returns
        -------
        data : dict
            Dictionary of data for a single session.
        """
        data = {"data": array}

        # Add other session labels
        placeholder = np.zeros(array.shape[0], dtype=np.float32)
        for session_label in self.session_labels:
            label_name = session_label.name
            label_values = session_label.values
            data[label_name] = placeholder + label_values[i]

        return data

    def dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
        drop_last_batch=False,
        repeat_count=1,
    ):
        """Create a Tensorflow Dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        shuffle : bool, optional
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        repeat_count : int, optional
            Number of times to repeat the dataset. Default is once.

        Returns
        -------
        dataset : tf.data.Dataset or tuple of tf.data.Dataset
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_size = step_size or sequence_length
        self.validation_split = validation_split

        # Validate batching
        self._validate_batching(
            sequence_length,
            batch_size,
            step_size=self.step_size,
            drop_last_batch=drop_last_batch,
            concatenate=concatenate,
        )

        n_sequences = self.count_sequences(self.sequence_length)

        def _create_dataset(X, shuffle=shuffle, repeat_count=repeat_count):
            # X is a list of np.ndarray

            # Create datasets for each array
            datasets = []
            for i in range(len(X)):
                data = self._create_data_dict(i, X[i])
                dataset = dtf.create_dataset(
                    data,
                    self.sequence_length,
                    self.step_size,
                )
                datasets.append(dataset)

            # Create a dataset from all the arrays concatenated
            if concatenate:
                if shuffle:
                    # Do a perfect shuffle then concatenate across arrays
                    random.shuffle(datasets)
                    full_dataset = dtf.concatenate_datasets(datasets)

                    # Shuffle sequences
                    full_dataset = full_dataset.shuffle(self.buffer_size)

                    # Group into mini-batches
                    full_dataset = full_dataset.batch(
                        self.batch_size, drop_remainder=drop_last_batch
                    )

                    # Shuffle mini-batches
                    full_dataset = full_dataset.shuffle(self.buffer_size)

                else:
                    # Concatenate across arrays
                    full_dataset = dtf.concatenate_datasets(datasets)

                    # Group into mini-batches
                    full_dataset = full_dataset.batch(
                        self.batch_size, drop_remainder=drop_last_batch
                    )

                # Repeat the dataset
                full_dataset = full_dataset.repeat(repeat_count)

                import tensorflow as tf  # moved here to avoid slow imports

                return full_dataset.prefetch(tf.data.AUTOTUNE)

            # Otherwise create a dataset for each array separately
            else:
                full_datasets = []
                for i, ds in enumerate(datasets):
                    if shuffle:
                        # Shuffle sequences
                        ds = ds.shuffle(self.buffer_size)

                    # Group into batches
                    ds = ds.batch(self.batch_size, drop_remainder=drop_last_batch)

                    if shuffle:
                        # Shuffle batches
                        ds = ds.shuffle(self.buffer_size)

                    # Repeat the dataset
                    ds = ds.repeat(repeat_count)

                    import tensorflow as tf  # moved here to avoid slow imports

                    full_datasets.append(ds.prefetch(tf.data.AUTOTUNE))

                return full_datasets

        # Trim data to be an integer multiple of the sequence length
        X = []
        for i in range(self.n_sessions):
            if i not in self.keep:
                # We don't want to include this session
                continue
            x = self.arrays[i]
            n = n_sequences[i]
            X.append(x[: n * sequence_length])

        if validation_split is not None:
            # Number of sequences that should be in the validation set
            n_val_sequences = (validation_split * n_sequences).astype(int)

            if np.all(n_val_sequences == 0):
                raise ValueError(
                    "No full sequences could be assigned to the validation set. "
                    "Consider reducing the sequence_length."
                )

            X_train = []
            X_val = []
            for i in range(self.n_sessions):
                if i not in self.keep:
                    continue

                # Randomly pick sequences
                val_indx = np.random.choice(
                    n_sequences[i], size=n_val_sequences[i], replace=False
                )
                train_indx = np.setdiff1d(np.arange(n_sequences[i]), val_indx)

                # Split data
                x = X[i].reshape(-1, sequence_length, self.n_channels)
                x_train = x[train_indx].reshape(-1, self.n_channels)
                x_val = x[val_indx].reshape(-1, self.n_channels)
                X_train.append(x_train)
                X_val.append(x_val)

            return _create_dataset(X_train), _create_dataset(
                X_val, shuffle=False, repeat_count=1
            )

        else:
            return _create_dataset(X)

    def save_tfrecord_dataset(
        self,
        tfrecord_dir,
        sequence_length,
        step_size=None,
        validation_split=None,
        overwrite=False,
    ):
        """Save the data as TFRecord files.

        Parameters
        ----------
        tfrecord_dir : str
            Directory to save the TFRecord datasets.
        sequence_length : int
            Length of the segement of data to feed into the model.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        overwrite : bool, optional
            Should we overwrite the existing TFRecord datasets if there is a need?
        """
        os.makedirs(tfrecord_dir, mode=0o700, exist_ok=True)

        self.sequence_length = sequence_length
        self.step_size = step_size or sequence_length
        self.validation_split = validation_split

        def _check_rewrite():
            if not os.path.exists(f"{tfrecord_dir}/tfrecord_config.pkl"):
                _logger.warning(
                    "No tfrecord_config.pkl file found. Rewriting TFRecords."
                )
                return True

            if not overwrite:
                return False

            # Check if we need to rewrite the TFRecord datasets
            tfrecord_config = misc.load(f"{tfrecord_dir}/tfrecord_config.pkl")

            if tfrecord_config["identifier"] != self._identifier:
                _logger.warning("Identifier has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["sequence_length"] != self.sequence_length:
                _logger.warning("Sequence length has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["n_channels"] != self.n_channels:
                _logger.warning("Number of channels has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["step_size"] != self.step_size:
                _logger.warning("Step size has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["validation_split"] != self.validation_split:
                _logger.warning("Validation split has changed. Rewriting TFRecords.")
                return True

            for label in self.session_labels:
                if label.name not in tfrecord_config["session_labels"]:
                    _logger.warning(
                        f"Session label {label} not found. Rewriting TFRecords."
                    )
                    return True

            return False

        # Number of sequences
        n_sequences = self.count_sequences(self.sequence_length)
        if validation_split is not None:
            n_val_sequences = (validation_split * n_sequences).astype(int)
            if np.all(n_val_sequences == 0):
                raise ValueError(
                    "No full sequences could be assigned to the validation set. "
                    "Consider reducing the sequence_length."
                )

        # Path to TFRecord file
        tfrecord_path = (
            f"{tfrecord_dir}"
            "/dataset-{val}_{array:0{v}d}-of-{n_session:0{v}d}"
            f".{self._identifier}.tfrecord"
        )

        # TFRecords we need to save
        tfrecord_filenames = []
        tfrecords_to_save = []
        rewrite = _check_rewrite()
        for i in self.keep:
            filepath = tfrecord_path.format(
                array=i,
                val="{val}",
                n_session=self.n_sessions - 1,
                v=len(str(self.n_sessions - 1)),
            )
            tfrecord_filenames.append(filepath)

            rewrite_ = rewrite or not os.path.exists(filepath.format(val=0))
            if validation_split is not None:
                rewrite_ = rewrite_ or not os.path.exists(filepath.format(val=1))
            if rewrite_:
                tfrecords_to_save.append((i, filepath))

        # Function for saving a single TFRecord
        def _save_tfrecord(i, filepath):
            # Trim data to be an integer multiple of the sequence length
            x = self.arrays[i][: n_sequences[i] * sequence_length]

            if validation_split is not None:
                # Randomly pick sequences
                val_indx = np.random.choice(
                    n_sequences[i], size=n_val_sequences[i], replace=False
                )
                train_indx = np.setdiff1d(np.arange(n_sequences[i]), val_indx)

                # Split data
                x = x.reshape(-1, sequence_length, self.n_channels)
                x_train = x[train_indx].reshape(-1, self.n_channels)
                x_val = x[val_indx].reshape(-1, self.n_channels)

                # Save datasets
                X_train = self._create_data_dict(i, x_train)
                dtf.save_tfrecord(
                    X_train,
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=0),
                )
                X_val = self._create_data_dict(i, x_val)
                dtf.save_tfrecord(
                    X_val,
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=1),
                )

            else:
                # Save the dataset
                X = self._create_data_dict(i, x)
                dtf.save_tfrecord(
                    X,
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=0),
                )

        # Save TFRecords
        if len(tfrecords_to_save) > 0:
            pqdm(
                array=tfrecords_to_save,
                function=_save_tfrecord,
                n_jobs=self.n_jobs,
                desc="Creating TFRecord datasets",
                argument_type="args",
                total=len(tfrecords_to_save),
            )

        # Save tfrecords config
        if rewrite:
            tfrecord_config = {
                "identifier": self._identifier,
                "sequence_length": self.sequence_length,
                "n_channels": self.n_channels,
                "step_size": self.step_size,
                "validation_split": self.validation_split,
                "session_labels": [label.name for label in self.session_labels],
                "n_sessions": self.n_sessions,
            }
            misc.save(f"{tfrecord_dir}/tfrecord_config.pkl", tfrecord_config)

    def tfrecord_dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
        drop_last_batch=False,
        repeat_count=1,
        tfrecord_dir=None,
        overwrite=False,
    ):
        """Create a TFRecord Dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        shuffle : bool, optional
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        repeat_count : int, optional
            Number of times to repeat the dataset. Default is once.
        tfrecord_dir : str, optional
            Directory to save the TFRecord datasets. If :code:`None`, then
            :code:`Data.store_dir` is used.
        overwrite : bool, optional
            Should we overwrite the existing TFRecord datasets if there is a need?

        Returns
        -------
        dataset : tf.data.TFRecordDataset or tuple of tf.data.TFRecordDataset
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
        """
        tfrecord_dir = tfrecord_dir or self.store_dir

        # Validate batching
        self._validate_batching(
            sequence_length,
            batch_size,
            step_size=(step_size or sequence_length),
            drop_last_batch=drop_last_batch,
            concatenate=concatenate,
        )

        # Save and load the TFRecord files
        self.save_tfrecord_dataset(
            tfrecord_dir=tfrecord_dir,
            sequence_length=sequence_length,
            step_size=step_size,
            validation_split=validation_split,
            overwrite=overwrite,
        )
        return dtf.load_tfrecord_dataset(
            tfrecord_dir=tfrecord_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            concatenate=concatenate,
            drop_last_batch=drop_last_batch,
            repeat_count=repeat_count,
            buffer_size=self.buffer_size,
            keep=self.keep,
        )

    def add_session_labels(self, label_name, label_values, label_type):
        """Add session labels as a new channel to the data.

        Parameters
        ----------
        label_name : str
            Name of the new channel.
        label_values : np.ndarray
            Labels for each session.
        label_type : str
            Type of label, either "categorical" or "continuous".
        """
        if len(label_values) != self.n_sessions:
            raise ValueError(
                "label_values must have the same length as the number of sessions."
            )

        self.session_labels.append(SessionLabels(label_name, label_values, label_type))

    def get_session_labels(self):
        """Get the session labels.

        Returns
        -------
        session_labels : List[SessionLabels]
            List of session labels.
        """
        return self.session_labels

    def save_preparation(self, output_dir="."):
        """Save a pickle file containing preparation settings.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        attributes = list(self.__dict__.keys())
        dont_keep = [
            "_identifier",
            "data_field",
            "picks",
            "reject_by_annotation",
            "original_sampling_frequency",
            "sampling_frequency",
            "mask_file",
            "parcellation_file",
            "time_axis_first",
            "load_memmaps",
            "buffer_size",
            "n_jobs",
            "prepared_data_filenames",
            "inputs",
            "store_dir",
            "raw_data_arrays",
            "raw_data_filenames",
            "n_raw_data_channels",
            "arrays",
            "keep",
            "use_tfrecord",
        ]
        for item in dont_keep:
            if item in attributes:
                attributes.remove(item)
        preparation = {a: getattr(self, a) for a in attributes}
        pickle.dump(preparation, open(f"{output_dir}/preparation.pkl", "wb"))

    def load_preparation(self, inputs):
        """Loads a pickle file containing preparation settings.

        Parameters
        ----------
        inputs : str
            Path to directory containing the pickle file with preparation
            settings.
        """
        if os.path.isdir(inputs):
            for file in rw.list_dir(inputs):
                if "preparation.pkl" in file:
                    preparation = pickle.load(open(f"{inputs}/preparation.pkl", "rb"))
                    for attr, value in preparation.items():
                        setattr(self, attr, value)
                    break

    def save(self, output_dir="."):
        """Saves (prepared) data to numpy files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        # Create output directory
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Function to save a single array
        def _save(i, arr):
            padded_number = misc.leading_zeros(i, self.n_sessions)
            np.save(f"{output_dir}/array{padded_number}.npy", arr)

        # Save arrays in parallel
        pqdm(
            enumerate(self.arrays),
            _save,
            desc="Saving data",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )

        # Save preparation settings
        self.save_preparation(output_dir)

    def delete_dir(self):
        """Deletes :code:`store_dir`."""
        if self.store_dir.exists():
            rmtree(self.store_dir)


@dataclass
class SessionLabels:
    """Class for session labels.

    Parameters
    ----------
    name : str
        Name of the session label.
    values : np.ndarray
        Value for each session. Must be a 1D array of numbers.
    label_type : str
        Type of the session label. Options are "categorical" and "continuous".
    """

    name: str
    values: np.ndarray
    label_type: str

    def __post_init__(self):
        if self.label_type not in ["categorical", "continuous"]:
            raise ValueError("label_type must be 'categorical' or 'continuous'.")

        if self.values.ndim != 1:
            raise ValueError("values must be a 1D array.")

        if self.label_type == "categorical":
            self.values = self.values.astype(np.int32)
            self.n_classes = len(np.unique(self.values))
        else:
            self.values = self.values.astype(np.float32)
            self.n_classes = None
