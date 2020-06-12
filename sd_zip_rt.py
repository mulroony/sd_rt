print("Staring up....\n\n")
print("Loading libraries...")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt

from datetime import date
from datetime import datetime

# Quiet pymc3
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)

import yaml
import pprint

print("    - done\n==================================")
print("Setup some functions and classes...")

def confirmed_to_onset(confirmed, p_delay):

    assert not confirmed.isna().any()
    
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    return onset

def adjust_onset_for_right_censorship(onset, p_delay):
    cumulative_p_delay = p_delay.cumsum()
    
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    
    return adjusted, cumulative_p_delay

class MCMCModel(object):
    
    def __init__(self, region, onset, cumulative_p_delay, window=50):
        
        # Just for identification purposes
        self.region = region
        
        # For the model, we'll only look at the last N
        self.onset = onset.iloc[-window:]
        self.cumulative_p_delay = cumulative_p_delay[-window:]
        
        # Where we store the results
        self.trace = None
        self.trace_index = self.onset.index[1:]

    def run(self, chains=1, tune=3000, draws=1000, target_accept=.95):

        with pm.Model() as model:

            # Random walk magnitude
            step_size = pm.HalfNormal('step_size', sigma=.03)

            # Theta random walk
            theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
            theta_raw_steps = pm.Normal('theta_raw_steps', shape=len(self.onset)-2) * step_size
            theta_raw = tt.concatenate([[theta_raw_init], theta_raw_steps])
            theta = pm.Deterministic('theta', theta_raw.cumsum())

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta/gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            self.trace = pm.sample(
                chains=chains,
                tune=tune,
                draws=draws,
                target_accept=target_accept)
            
            return self
    
    def run_gp(self):
        with pm.Model() as model:
            gp_shape = len(self.onset) - 1

            length_scale = pm.Gamma("length_scale", alpha=3, beta=.4)

            eta = .05
            cov_func = eta**2 * pm.gp.cov.ExpQuad(1, length_scale)

            gp = pm.gp.Latent(mean_func=pm.gp.mean.Constant(c=0), 
                              cov_func=cov_func)

            # Place a GP prior over the function f.
            theta = gp.prior("theta", X=np.arange(gp_shape)[:, None])

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta / gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            self.trace = pm.sample(chains=1, tune=1000, draws=1000, target_accept=.8)
        return self


def df_from_model(model):
    
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    #hpd_90 = pm.stats.hpd(r_t, credible_interval=.9)
    #hpd_50 = pm.stats.hpd(r_t, credible_interval=.5)
    hpd_90 = pm.stats.hpd(r_t, hdi_prob=.9)
    hpd_50 = pm.stats.hpd(r_t, hdi_prob=.5)
    
    idx = pd.MultiIndex.from_product([
            [model.region],
            model.trace_index
        ], names=['region', 'date'])
        
    df = pd.DataFrame(data=np.c_[mean, median, hpd_90, hpd_50], index=idx,
                 columns=['mean', 'median', 'lower_90', 'upper_90', 'lower_50','upper_50'])
    return df

def create_and_run_model(name, state):
    confirmed = state.positive.diff().dropna()
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return MCMCModel(name, onset, cumulative_p_delay).run()

print("    - done\n==================================")
print("Setup config...")

#####################
# Defaults
#####################
config = {}
config['dates_to_do'] = []
config['zipcodes_to_do'] = []
config['zip_url'] = 'https://opendata.arcgis.com/datasets/854d7e48e3dc451aa93b9daf82789089_0.csv'
config['output_path'] = '/covid/outputs/sd_zip_rt.xlsx'
config['patient_path'] = '/covid/inputs/patients.20200611.csv'
#####################

print("    - done\n==================================")
print("Read YAML Config...")

with open("/covid/inputs/config.yaml") as file:
    yaml_config = yaml.full_load(file)
    for key, value in yaml_config.items():
        config[key] = value

# Convert to timestamps
if config['dates_to_do']:
    config['dates_to_do'] = pd.to_datetime(config['dates_to_do'])
else:
    config['dates_to_do'] = pd.to_datetime([])

# Zip codes need to be integers
if config['zipcodes_to_do']:
    config['zipcodes_to_do'] = [int(x) for x in config['zipcodes_to_do'] ]


print("    - done\n==================================")
print("Configuration:")

pprint.pprint(config)

print("\n==================================")
print("Reading remote CSV...")

zips = pd.read_csv(config['zip_url'])

print("    - done\n==================================")
print("Clean up dataframe...")

zips['date'] = zips['updatedate'].apply(lambda x: pd.to_datetime(x.split(" ")[0]))

zips.drop(inplace=True, columns=['X',
                                 'Y',
                                 'FID',
                                 'zipcode_zip',
                                 'created_date',
                                 'updatedate',
                                 'created_date', 
                                 'created_user', 
                                 'last_edited_date', 
                                 'last_edited_user',
                                 'globalid'])

zips.rename(columns={'ziptext': 'zipcode','case_count': 'positive'}, inplace=True)

zips = zips.set_index(['zipcode', 'date']).sort_index()

print("    - done\n==================================")
print("Reading patient CSV...")
patients = pd.read_csv(config['patient_path'],
                       parse_dates=['Onset', 'Confirmed'],
                       low_memory=False)

print("    - done\n==================================")
print("Calculating p_delay...")

# Calculate the delta in days between onset and confirmation
delay = (patients.Confirmed - patients.Onset).dt.days

# Convert samples to an empirical distribution
p_delay = delay.value_counts().sort_index()
new_range = np.arange(0, p_delay.index.max()+1)
p_delay = p_delay.reindex(new_range, fill_value=0)
p_delay /= p_delay.sum()

print("    - done\n==================================")
print("Running Model code for each zip, this takes a while...\n")

models = {}

for zipcode, grp in zips.groupby('zipcode'):
   
    # Skip zipcodes not in config
    if config['zipcodes_to_do'] and zipcode not in config['zipcodes_to_do']:
        print(f'    - Skipping {zipcode}, not in list to do')
        continue
 
    print(zipcode)
    if zipcode in models:
        print(f'    - Skipping {zipcode}, already in cache')
        continue
    
    if zips.loc[zipcode].isnull().all().bool():
        print(f'    - Skipping {zipcode}, all values null')
        continue
   
    print(f'    - {zipcode}....')
    try: 
        models[zipcode] = create_and_run_model(zipcode, grp.droplevel(0))
    except:
        print(f'!!ERROR - Model code failed, skipping...')

print("    - done\n==================================")
print("Checking for divergences, not sure this does anything...")

# Check to see if there were divergences
n_diverging = lambda x: x.trace['diverging'].nonzero()[0].size
divergences = pd.Series([n_diverging(m) for m in models.values()], index=models.keys())
has_divergences = divergences.gt(0)

if divergences[has_divergences].empty:
    print('No divergences found...')
else:
    print('Found divergences... they will rerun now')

# Rerun zipcodes with divergences
for zipcode, n_divergences in divergences[has_divergences].items():
    print(f'    - Running {zipcode} again due to divergence')
    models[zipcode].run()

print("    - done\n==================================")
print("Converting results to dataframe")

results = None


print("Converting model results to dataframe")
for zipcode, model in models.items():

    df = df_from_model(model)

    if results is None:
        results = df
    else:
        results = pd.concat([results, df], axis=0)



print("    - done\n==================================")
print("Writing output to %s..."%(config['output_path']))

with pd.ExcelWriter(config['output_path']) as writer:
    for i in pd.unique(results.reset_index()['region']):
        if config['dates_to_do'].empty:
            results.loc[i].to_excel(writer, sheet_name=str(i))
        else:
            results.loc[i].loc[config['dates_to_do']].to_excel(writer, sheet_name=str(i))

print("    - done\n==================================")
print("That's all folks...")
