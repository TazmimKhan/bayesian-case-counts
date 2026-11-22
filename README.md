# Bayesian Modelling of Infectious-Disease Case Counts

**Question:** How can Bayesian methods estimate trends in reported disease cases,
and how certain are those estimates? What can reported counts *not* tell us?

## What the project does
1. Tests the model on simulated data with known parameters (parameter recovery).
2. Fits a log-linear **Poisson** model to weekly COVID-19 cases in Bangladesh
   (first growth phase, late April to late June 2020) using MCMC in PyMC.
3. Checks convergence with trace plots, R-hat, effective sample size and divergences.
4. Uses posterior predictive checks to show the Poisson model is overconfident,
   then fits a **Negative Binomial** model and compares the two with LOO.
5. Adds a **detection probability** to show that true infections cannot be
   identified from reported counts alone, while the growth rate still can,
   plus a sensitivity analysis.

## Data
Our World in Data COVID-19 dataset
(`https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv`),
daily new cases aggregated to weekly totals. Downloaded on: **17 November 2026**.
The first run downloads about 185 MB and saves the chosen country to `data/cases_<Country>.csv`.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook bayesian_case_counts.ipynb
```
Or run `python bayesian_case_counts.py`. Figures are saved in `figures/<Country>/`.

## Key results  *(write in your own words after running)*
- Weekly growth factor (Negative Binomial): **[…]** (95% credible interval […])
- Doubling time: **[…]** weeks
- Model comparison: **[…]**
- Detection probability: the posterior of *p* is almost identical to its prior → not identifiable.

![Posterior predictive check, Negative Binomial model](figures/Bangladesh/05_ppc_negbin.png)

![Detection probability is not identifiable](figures/Bangladesh/07_identifiability.png)

## Limitations
- A single log-linear trend fits only one growth phase.
- Reported counts depend on testing volume and reporting delays.
- Few data points, so LOO is unstable (Pareto k̂ warnings).
- True infections need extra information (serology, testing data, or multiple
  data sources as in capture-recapture).

## Tools
Python, pandas, NumPy, SciPy, PyMC, ArviZ, Matplotlib.

## Acknowledgement
Code developed with the assistance of AI tools; data analysis, interpretation,
and write-up are my own.
