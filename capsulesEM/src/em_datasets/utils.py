"""Utilities functions for loading datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import time

from six.moves import urllib
from tensorflow.python.platform import gfile


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
  """Simple decorator for wrapping retriable functions.
  Args:
    initial_delay: the initial delay.
    max_delay: the maximum delay allowed, actual max is
        max_delay * (1 + jitter).
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.
  """
  if factor < 1:
    raise ValueError('factor must be >= 1; was %f' % (factor,))

  if jitter >= 1:
    raise ValueError('jitter must be < 1; was %f' % (jitter,))

  # Generator to compute the individual delays
  def delays():
    delay = initial_delay
    while delay <= max_delay:
      yield delay * random.uniform(1 - jitter, 1 + jitter)
      delay *= factor

  def wrap(fn):
    """Wrapper function factory invoked by decorator magic."""

    def wrapped_fn(*args, **kwargs):
      """The actual wrapper function that applies the retry logic."""
      for delay in delays():
        try:
          return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except)
          if is_retriable is None:
            continue

          if is_retriable(e):
            time.sleep(delay)
          else:
            raise
      return fn(*args, **kwargs)

    return wrapped_fn

  return wrap


_RETRIABLE_ERRNOS = {
  110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
  return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
  return urllib.request.urlretrieve(url, filename)


def download(filename, work_directory, source_url, overwrite=False):
  """Download the data from source url.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist, or overwrite is True.
      overwrite: boolean, if True, download and overwrite current file even when the file is exist.

  Returns:
      Path to the resulting file.
  """

  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)

  filepath = os.path.join(work_directory, filename)

  if overwrite or not gfile.Exists(filepath):
    _filename, _ = urlretrieve_with_retry(source_url + filename)
    print('_filename:', _filename)
    gfile.Copy(_filename, filepath, overwrite=overwrite)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')

  return filepath

