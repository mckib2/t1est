T1 fitting
----------

Simple T1 fitting.  Uses the following fitting model:
x = A - B \exp{-t/T_1}

where A and B are complex coefficients, t are the inversion times.

Installation
------------

Should be as easy as:

.. code-block:: bash

    pip install t1est

Usage
-----

Run examples as follows:

.. code-block:: python

    python -m t1est.examples.basic_usage

The function `t1est()` can be called as follows:

.. code-block:: python

    from t1est import t1est

    # x is the array of images at different inversion times.  The
    # inversion times are provided as TIs
    T1map = t1est(
        x, TIs, time_axis=-1, mask=mask, method='trf', T1_bnds=(0, 3),
        chunksize=10, molli=True, mag=True)

Notice that `x` may be any-dimensional, but time points must lie
along the `time_axis` dimension. If `molli=True`, then T1 is adjusted
as follows: T1' = T1 (B/A - 1)
