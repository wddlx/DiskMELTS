# Uncertainty API

`diskmelts.uncertainty` turns a best fit into posterior error bars. The model
and grid fitters call it automatically; the functions are public for custom
likelihoods. See {ref}`uncertainties` for how to read the results.

```{eval-rst}
.. automodule:: diskmelts.uncertainty
   :no-members:

.. autofunction:: diskmelts.uncertainty.sample_posterior

.. autofunction:: diskmelts.uncertainty.likelihood_sigma

.. autofunction:: diskmelts.uncertainty.noise_correlation_length

.. autofunction:: diskmelts.uncertainty.summarize_parameter

.. autofunction:: diskmelts.uncertainty.split_r_hat

.. autofunction:: diskmelts.uncertainty.default_mcmc_steps
```
