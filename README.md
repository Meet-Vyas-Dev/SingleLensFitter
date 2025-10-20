# SingleLensFitter

See the wiki for some basic documentation.

-----

# SingleLensFitter

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://www.google.com/search?q=)

`SingleLensFitter` is a Python class for modeling and analyzing single-lens gravitational microlensing events. It provides a flexible, modular framework for fitting photometric data using a Bayesian approach with Markov Chain Monte Carlo (MCMC) sampling.

## Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Quickstart Example](https://www.google.com/search?q=%23quickstart-example)
  - [Documentation](https://www.google.com/search?q=%23documentation)
      - [1. Class Initialization](https://www.google.com/search?q=%231-class-initialization)
      - [2. Model Configuration Methods](https://www.google.com/search?q=%232-model-configuration-methods)
      - [3. Fitting and Sampling](https://www.google.com/search?q=%233-fitting-and-sampling)
      - [4. Output Files](https://www.google.com/search?q=%234-output-files)
      - [5. Public Attributes](https://www.google.com/search?q=%235-public-attributes)
      - [6. Core Internal Methods](https://www.google.com/search?q=%236-core-internal-methods)

## Features

  * **Modular Model Building**: Start with a simple Point-Source, Point-Lens (PSPL) model and easily add complexity.
  * **Advanced Effects**: Includes support for finite source effects, linear limb-darkening, and sinusoidal variability in the source or blend.
  * **Robust Fitting**: Implements a Gaussian Process (GP) model for correlated noise and a mixture model for robust outlier rejection.
  * **Efficient MCMC Sampling**: Uses `emcee` for Bayesian parameter estimation and analytically marginalizes linear parameters (source/blend flux) for faster convergence.
  * **Publication-Ready Plots**: Automatically generates light curve fits, residual plots, and corner plots of the posterior distributions.
  * **Multi-Dataset Fitting**: Simultaneously fit data from multiple observatories.

## Dependencies

The class requires the following Python libraries:

  * `numpy`
  * `scipy`
  * `emcee`
  * `george`
  * `matplotlib`
  * `corner`
  * `astropy`
  * `mpmath`

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/SingleLensFitter.git
    cd SingleLensFitter
    ```

2.  **Install the required packages:**
    It is highly recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    Where `requirements.txt` contains:

    ```
    numpy
    scipy
    emcee
    george
    matplotlib
    corner
    astropy
    mpmath
    ```

## Quickstart Example

Here is a simple example of how to load data, configure a model, and run the MCMC sampler.

```python
import numpy as np
import matplotlib.pyplot as plt
# Assuming the class is in a file named 'SingleLensFitter.py'
from SingleLensFitter import SingleLensFitter

# 1. Load your data
# Create some dummy data for this example
# In a real case, you would load your data from a file
t0_true, u0_true, tE_true = 7500.0, 0.1, 30.0
true_params = [u0_true, t0_true, tE_true]
time = np.linspace(t0_true - 2 * tE_true, t0_true + 2 * tE_true, 500)
tau = (time - t0_true) / tE_true
u = np.sqrt(u0_true**2 + tau**2)
magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
flux = 1000 * magnification + 500  # F_source = 1000, F_blend = 500
flux_err = np.random.normal(0, 20, len(time))  # Add some noise
flux += flux_err
err = np.ones_like(flux) * 20.0

data_dict = {'MyObservatory': (time, flux, err)}

# 2. Set initial parameter guesses: [u0, t0, tE]
initial_params = np.array([0.15, 7505.0, 25.0])

# 3. Initialize the fitter
fitter = SingleLensFitter(data=data_dict, initial_parameters=initial_params)

# 4. (Optional) Customize the model
# Add finite source and limb darkening effects
fitter.add_limb_darkening(gamma=0.5, lrho=-2.5)
# Add a Gaussian Process model for correlated noise
fitter.add_gaussian_process_model()

# 5. Customize fitter settings
fitter.plotprefix = 'ob12345_fit'
fitter.nwalkers = 100
fitter.nsteps_production = 1000 # Reduced for a quick example

# 6. Run the MCMC sampler
# optimize_first=True is recommended to start walkers near the peak likelihood
fitter.sample(optimize_first=True)

# 7. The results are now stored in fitter.p and fitter.samples,
#    and plots have been saved to disk.
print("Fit results:")
print(f"Best-fit parameters (50th percentile): {fitter.p}")

# 8. Display one of the output plots
plt.figure()
img = plt.imread('ob12345_fit-combined-lightcurve.png')
plt.imshow(img)
plt.axis('off')
plt.show()


```

-----

## Documentation

### 1\. Class Initialization

#### `SingleLensFitter(data, initial_parameters, eigen_lightcurves=None, reference_source=None, ZP=28.0)`

Initializes the fitter object with data and initial model parameters.

  - **`data`** (`dict`): A dictionary where each key is a string identifying a dataset (e.g., the observatory name) and each value is a tuple of three `numpy` arrays: `(time, flux, flux_error)`.

      - *Example*: `{'OGLE': (hjd, flux, flux_err), 'MOA': (hjd_moa, flux_moa, flux_err_moa)}`

  - **`initial_parameters`** (`numpy.ndarray`): An array containing the initial guess for the fundamental microlensing parameters in the following order:

    1.  `u_0`: The lens-source impact parameter in units of the Einstein radius ($R_E$).
    2.  `t_0`: The time of closest approach (HJD - 2450000).
    3.  `t_E`: The Einstein radius crossing time in days.

  - **`eigen_lightcurves`** (`dict`, optional): A dictionary for linear detrending using pre-computed basis vectors (e.g., from PCA). The keys should match the keys in the `data` dictionary. Each value should be a 2D `numpy` array of shape `(n_vectors, n_datapoints)`, where each row is a detrending vector. Defaults to `None`.

  - **`reference_source`** (`str`, optional): The key from the `data` dictionary corresponding to the dataset that will be used as the reference for magnitude normalization when creating combined light curve plots. If `None`, the first dataset is chosen automatically. Defaults to `None`.

  - **`ZP`** (`float`, optional): The magnitude zero-point to use for converting fluxes to magnitudes in plots. Defaults to `28.0`.

### 2\. Model Configuration Methods

These methods add complexity to the standard PSPL model. They must be called *before* running the fit.

#### `add_finite_source(lrho=None)`

Includes finite source effects in the model. This adds the parameter $\rho = R_* / R_E$ (the source radius in units of the Einstein radius) to the fit. The parameter is fit in log-space, $log_{10}(\rho)$.

  - **`lrho`** (`float`, optional): An initial guess for $log_{10}(\rho)$. If `None`, it defaults to `-3.0`.

#### `add_limb_darkening(gamma=None, lrho=None)`

Includes linear limb-darkening in the model. This automatically enables finite source effects if they haven't been added already.

  - **`gamma`** (`float`, optional): An initial guess for the linear limb-darkening coefficient, $\Gamma$. Defaults to `0.1`.
  - **`lrho`** (`float`, optional): An initial guess for $log_{10}(\rho)$, only used if `add_finite_source` has not been called previously.

#### `add_source_variability(params=None)`

Models intrinsic sinusoidal variability of the source star by adding three parameters: amplitude (`K`), angular frequency (`\omega`), and phase (`\phi`).

  - **`params`** (`tuple`, optional): Initial guesses for `(K, omega, phi)`. Defaults to `(0.001, np.pi, 0.0)`.

#### `add_blend_variability(params=None)`

Models sinusoidal variability of the blend flux. The parameters are the same as for `add_source_variability`.

  - **`params`** (`tuple`, optional): Initial guesses for `(K, omega, phi)`. Defaults to `(0.001, np.pi, 0.0)`.

#### `add_mixture_model()`

Adds a mixture model to account for outlier data points. For each dataset, this adds three parameters: `P_b` (outlier probability), `V_b` (outlier variance), and `Y_b` (outlier mean).

#### `add_gaussian_process_model(common=True)`

Models correlated noise using a Gaussian Process with an `ExpKernel`. This adds two parameters per dataset: `ln_a` (log amplitude) and `ln_tau` (log length-scale).

  - **`common`** (`bool`, optional): If `True`, a single set of GP hyperparameters is fit for all datasets. Defaults to `True`.

### 3\. Fitting and Sampling

#### `fit(method='Nelder-Mead')`

Performs a maximum likelihood optimization to find the best-fit parameters using `scipy.optimize.minimize`. This is useful for finding a good starting point for MCMC walkers.

  - **`method`** (`str`, optional): The optimization algorithm to use. Defaults to `'Nelder-Mead'`.

#### `sample(optimize_first=False)`

The main execution method. It runs an MCMC simulation to sample the posterior probability distribution of the fitted parameters. The process involves an automated burn-in phase with convergence checks, followed by a production run.

  - **`optimize_first`** (`bool`, optional): If `True`, runs the `fit()` method before starting the MCMC sampler. Defaults to `False`.

### 4\. Output Files

The `sample()` method generates several output files, prefixed with the string stored in the `plotprefix` attribute:

  - `{plotprefix}.fit_results`: A text file summarizing the median value and 16th/84th percentile uncertainties for each fitted parameter.
  - `{plotprefix}-lc.png`: A plot showing the individual light curve for each dataset with model fits from the posterior overlaid.
  - `{plotprefix}-combined-lightcurve.png`: A plot showing all datasets normalized and combined into a single light curve and residuals panel.
  - `{plotprefix}-pdist.png`: A corner plot showing the 1D and 2D projections of the posterior probability distributions.
  - `{plotprefix}-burnin.png` / `-final.png`: Plots of the MCMC chains for each parameter, useful for diagnosing convergence.
  - `{plotprefix}-state-burnin.npy` / `-production.npy`: NumPy binary files saving the final state of the MCMC walkers.

### 5\. Public Attributes

After running `sample()`, the following attributes are populated with results:

  - **`p`** (`numpy.ndarray`): The final best-fit parameters, taken as the 50th percentile (median) of the posterior distributions.
  - **`samples`** (`numpy.ndarray`): The flattened MCMC chain from the production run, with shape `(n_walkers * nsteps_production, n_dimensions)`. This array contains the full posterior samples.
  - **`parameter_labels`** (`list`): A list of strings containing LaTeX-formatted labels for each parameter, used for plotting.

### 6\. Core Internal Methods

While users do not call these methods directly, understanding their function is helpful.

  - **`lnprob(p)`**: The core function for `emcee`. It calculates the log-posterior probability for a given set of parameters `p`, defined as $ln(Posterior) = ln(Prior) + ln(Likelihood)$.
  - **`lnlikelihood()`**: Calculates the log-likelihood of the data given the model. It iterates through all datasets and sums their individual log-likelihoods.
  - **`lnprior_...()`**: A series of methods that define the priors for different parameters. They return `0.0` if a parameter is within its allowed range and `-np.inf` otherwise (a uniform prior).
  - **`magnification(t, p=None)`**: Computes the microlensing magnification at time(s) `t` for a given parameter set `p`. It handles the point-source case as well as numerical integration for finite source and limb-darkening models.
  - **`linear_fit(data_key, mag)`**: A crucial optimization step. For a given magnification model, this function analytically solves for the linear parameters (source flux $F_s$ and blend flux $F_b$) that maximize the likelihood. This process, known as marginalization, significantly improves the efficiency and convergence of the MCMC sampler.


If you use this code, please cite 

<a href="https://doi.org/10.5281/zenodo.265134"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.265134.svg" alt="DOI"></a>

