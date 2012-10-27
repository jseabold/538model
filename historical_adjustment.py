# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import datetime

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas
from scipy import stats
np.set_printoptions(precision=4, suppress=True)
pandas.set_printoptions(notebook_repr_html=False,
                        precision=4,
                        max_columns=12, column_space=10,
                        max_colwidth=25)
from matplotlib import rcParams
#rcParams['text.usetex'] = False
#rcParams['text.latex.unicode'] = False

# <markdowncell>

# We have a snapshot for what would happen if the election is held today (Don't go bet on intratrade based on this model). Historically, polls have narrowed as the election nears.

# <headingcell level=4>

# Set up some globals for dates

# <codecell>

today = datetime.datetime(2012, 10, 2)
election = datetime.datetime(2012, 11, 6)
days_before = election - today
date2004 = datetime.datetime(2004, 11, 2)
days_before2004 = date2004 - days_before
date2008 = datetime.datetime(2008, 11, 4)
days_before2008 = date2008 - days_before

# <headingcell level=3>

# TODO: Put a basemap map here

# <headingcell level=4>

# Load that data and clean

# <codecell>

national_2004 = pandas.read_table("/home/skipper/school/talks/538model/data/2004_poll_data.csv")
national_2004.rename(columns={"Poll" : "Pollster"}, inplace=True);

# <codecell>

state_data2004 = pandas.read_csv("/home/skipper/school/talks/538model/data/2004-pres-polls.csv")
state_data2008 = pandas.read_csv("/home/skipper/school/talks/538model/data/2008-pres-polls.csv")

# <codecell>

state_data2004

# <codecell>

state_data2004.rename(columns={"Kerry" : "challenger", 
                               "Bush" : "incumbent"}, 
                      inplace=True);
state_data2004["dem_spread"] = (state_data2004["challenger"] - 
                                      state_data2004["incumbent"])

# <codecell>

state_data2004.Date.replace({"Nov 00" : "Nov 01", "Oct 00" : "Oct 01"}, 
                            inplace=True);
state_data2004.Date = (state_data2004.Date + ", 2004").apply(
                                                pandas.datetools.parse)

# <codecell>

def median_date(row, year="2008"):
    dt1 = pandas.datetools.parse(row["Start"] + ", " + year)
    dt2 = pandas.datetools.parse(row["End"] + ", " + year)
    dates = pandas.date_range(dt1, dt2)
    median_idx = int(np.median(range(len(dates)))+.5)
    return dates[median_idx]

# <codecell>

state_data2008["Date"] = state_data2008.apply(median_date, axis=1)
del state_data2008["Start"]
del state_data2008["End"]

# <codecell>

actual = national_2004.head(1)
national_2004 = national_2004.ix[national_2004.index[~national_2004.Pollster.isin(["Final Results", "RCP Average"])]]

# <codecell>

def split_median_date(row):
    dt = row["Date"]
    dt1, dt2 = dt.split(" - ")
    dates = pandas.date_range(dt1 + ", 2004", dt2 + ", 2004")
    median_idx = int(np.median(range(len(dates)))+.5)
    return dates[median_idx]

# <codecell>

national_2004["Date"] = national_2004.apply(split_median_date, axis=1)

# <codecell>

national_2004["dem_spread"] = national_2004["Kerry (D)"] - national_2004["Bush (R)"]

# <codecell>

state_data2008

# <codecell>

state_data2008.rename(columns={"Obama" : "challenger", 
                               "McCain" : "incumbent"}, 
                      inplace=True);
state_data2008["dem_spread"] = (state_data2008["challenger"] - 
                                      state_data2008["incumbent"])

# <headingcell level=4>

# Clean the Pollster names

# <codecell>

import pickle
pollster_map = pickle.load(open(
                 "/home/skipper/school/talks/538model/data/pollster_map.pkl", "rb"))

# <codecell>

state_data2004.Pollster.replace(pollster_map, inplace=True);
state_data2008.Pollster.replace(pollster_map, inplace=True);
national_2004.Pollster.replace(pollster_map, inplace=True);

# <headingcell level=4>

# Get the Pollster weights

# <markdowncell>

# These are old weights obtained from the 538 web site. New weights are not published anywhere to my knowledge.

# <codecell>

weights = pandas.read_table("/home/skipper/school/talks/538model/"
                            "data/pollster_weights.csv")

# <codecell>

state_data2004 = state_data2004.merge(weights, on="Pollster", how="inner");
state_data2008 = state_data2008.merge(weights, on="Pollster", how="inner");

# <headingcell level=3>

# What's the Assertion?

# <codecell>

def edit_tick_label(tick_val, tick_pos):
    if tick_val  < 0:
        text = str(int(tick_val)).replace("-", "Republican+")
    else:
        text = "Democrat+"+str(int(tick_val))
    return text

# <codecell>

from pandas import lib
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(figsize=(12,8))

data = national_2004[["Date", "dem_spread"]]
#data = data.ix[data.Date >= days_before2004]
#data = pandas.concat((data, national_data2012[["Date", "dem_spread"]]))
    
data.sort("Date", inplace=True)
dates = pandas.DatetimeIndex(data.Date).asi8

x = data.dem_spread.values.astype(float)
lowess_res = sm.nonparametric.lowess(x, dates, 
                                    frac=.2, it=3)[:,1]

dates_x = lib.ints_to_pydatetime(dates)
axes.scatter(dates_x, data["dem_spread"])
axes.plot(dates_x, lowess_res, color='r', lw=4)
axes.yaxis.get_major_locator().set_params(nbins=12)
axes.yaxis.set_major_formatter(FuncFormatter(edit_tick_label))
axes.grid(False, axis='x')
axes.hlines(-1.21, dates_x[0], dates_x[-1], color='black', lw=3)
axes.vlines(datetime.datetime(2004, 8, 5), -20, 15, lw=3)
axes.margins(0, .00)

# <headingcell level=3>

# Let's look at the State Polls

# <codecell>

from pandas import lib
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(figsize=(12,8))

data = state_data2004[["Date", "dem_spread"]]
#data = data.ix[data.Date >= days_before2004]
data = data.ix[data.Date >= datetime.datetime(2004, 7, 15)]
#data = pandas.concat((data, national_data2012[["Date", "dem_spread"]]))
    
data.sort("Date", inplace=True)
dates = pandas.DatetimeIndex(data.Date).asi8

x = data.dem_spread.values.astype(float)
lowess_res = sm.nonparametric.lowess(x, dates, 
                                    frac=.2, it=3)[:,1]

dates_x = lib.ints_to_pydatetime(dates)
axes.scatter(dates_x, data["dem_spread"])
axes.plot(dates_x, lowess_res, color='r', lw=4)
axes.yaxis.get_major_locator().set_params(nbins=12)
axes.yaxis.set_major_formatter(FuncFormatter(edit_tick_label))
axes.grid(False, axis='x')
axes.hlines(-1.21, dates_x[0], dates_x[-1], color='black', lw=3)
axes.margins(0, .05)

# <codecell>

from pandas import lib
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(figsize=(12,8))

data = state_data2008[["Date", "dem_spread"]]
data = data.ix[data.Date >= datetime.datetime(2008, 7, 15)]
#data = data.ix[data.Date >= days_before2008]
#data = pandas.concat((data, national_data2012[["Date", "dem_spread"]]))
    
data.sort("Date", inplace=True)
dates = pandas.DatetimeIndex(data.Date).asi8

x = data.dem_spread.values.astype(float)
lowess_res = sm.nonparametric.lowess(x, dates, 
                                    frac=.2, it=3)[:,1]

dates_x = lib.ints_to_pydatetime(dates)
axes.scatter(dates_x, data["dem_spread"])
axes.plot(dates_x, lowess_res, color='r', lw=4)
axes.yaxis.get_major_locator().set_params(nbins=12)
axes.yaxis.set_major_formatter(FuncFormatter(edit_tick_label))
axes.grid(False, axis='x')
axes.hlines(3.65, dates_x[0], dates_x[-1], color='black', lw=3)
axes.vlines(datetime.datetime(2008, 8, 29), -45, 70, lw=3)
axes.vlines(datetime.datetime(2008, 9, 24), -45, 70, lw=3)
axes.margins(0, .0)

# <headingcell level=4>

# Clean the data

# <codecell>

#loadpy https://raw.github.com/gist/3912533/d958b515f602f6e73f7b16d8bc412bc8d1f433d9/state_abbrevs.py;

# <codecell>

states_abbrev_dict = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

# <codecell>

state_data2004.State.replace(states_abbrev_dict, inplace=True);
state_data2008.State.replace(states_abbrev_dict, inplace=True);

# <codecell>

state_data2004["days_until"] = date2004 - state_data2004.Date
state_data2008["days_until"] = date2008 - state_data2004.Date

# <codecell>

#state_data2004 = state_data2004.drop(
#                    state_data2004.index[state_data2004.days_until > days_before])
#state_data2008 = state_data2008.drop(
#                    state_data2008.index[state_data2008.days_until > days_before])

# <codecell>

def exp_decay(days):
    # defensive coding, accepts timedeltas
    days = getattr(days, "days", days)
    return .5 ** (days/30.)

# <codecell>

state_data2004["time_weight_oct2"] = (days_before2004 - 
                                      state_data2004["Date"]).apply(exp_decay)
state_data2004["time_weight_election"] = (date2004 -
                                      state_data2004["Date"]).apply(exp_decay)
state_data2008["time_weight_oct2"] = (days_before2008 - 
                                      state_data2008["Date"]).apply(exp_decay)
state_data2008["time_weight_election"] = (date2008 -
                                      state_data2008["Date"]).apply(exp_decay)

# <codecell>

def weighted_mean(group, weights_name):
    weights = group[weights_name]
    return np.sum(weights*group["dem_spread"]/np.sum(weights))

# <headingcell level=4>

# Get weighted average State-level polls for Oct 2 and Election Day

# <codecell>

def get_state_averages(dframe, time_weight_name):
    dframe_pollsters = dframe.groupby(["State", "Pollster"])
    dframe_result = dframe_pollsters.apply(weighted_mean, time_weight_name)
    dframe_result.name = "dem_spread"
    dframe_result = dframe_result.reset_index()
    dframe_result = dframe_result.merge(dframe[["Pollster", "Weight"]],
                          on="Pollster")
    return dframe_result.groupby("State").apply(weighted_mean, "Weight")
    

# <codecell>

oct2 = state_data2004.Date <= days_before2004
state_polls_oct2_2004 = get_state_averages(state_data2004.ix[oct2], "time_weight_oct2")
state_polls_election_2004 = get_state_averages(state_data2004, "time_weight_election")
updated2004 = state_data2004.ix[~oct2].State.unique()
updated2004.sort()

# <codecell>

oct2 = state_data2008.Date <= days_before2008
state_polls_oct2_2008 = get_state_averages(state_data2008.ix[oct2], "time_weight_oct2")
state_polls_election_2008 = get_state_averages(state_data2008, "time_weight_election")
updated2008 = state_data2008.ix[~oct2].State.unique()
updated2008.sort()

# <headingcell level=4>

# Get Economic Data

# <markdowncell>

# <table>
#     <thead>
#         <tr style="background: black; color: white; text-align: center">
#             <th style="padding: 15px; border-right-color: white; text-align: center">FRED Variable</th>
#             <th style="padding: 15px; border-left-color: white; text-align: center">Explanation</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td><b>PAYEMS</b></td>
#             <td>Nonfarm-Payrolls (Job Growth)</td>
#         </tr>
#         <tr>
#             <td><b>PI</b></td>
#             <td>Personal Income</td>
#         </tr>
#         <tr>
#             <td><b>INDPRO</b></td>
#             <td>Industrial Production</td>
#         </tr>
#         <tr>
#             <td><b>PCEC96</b></td>
#             <td>Consumption</td>
#         </tr>
#         <tr>
#             <td><b>CPIAUCSL</b></td>
#             <td>Inflation</td>
#         </tr>
#     </tbody>
# </table>

# <codecell>

from pandas.io.data import DataReader

# <codecell>

series = dict(jobs = "PAYEMS",
              income = "PI",
              prod = "INDPRO",
              cons = "PCEC96",
              prices = "CPIAUCSL")

# <codecell>

try:
    indicators = []
    for variable in series:
        data = DataReader(series[variable], "fred", start="2000-10-1")
        data.rename(columns={series[variable] : variable}, inplace=True)
        indicators.append(data)
    indicators = pandas.concat(indicators, axis=1)
    indicators.to_csv("/home/skipper/school/talks/538model/tmp_indicators_full.csv")
except: # probably not online
    indicators = pandas.read_csv("/home/skipper/school/talks/538model/tmp_indicators_full.csv", 
                                 parse_dates=True)
    indicators.set_index("DATE", inplace=True)
    # why doesn't it do this automaticall?
    indicators.index = pandas.DatetimeIndex(indicators.index)

# <markdowncell>

# For stock variables, just compute annualized quarterly growth rates (end - beginning)/beginning * 400 and average.

# <codecell>

quarterly_growth = np.log(indicators.resample("Q", 
                          how="mean")).diff() * 400
annualized = quarterly_growth.resample("A", how="mean")

# <codecell>

quarterly_growth = quarterly_growth.dropna()

# <markdowncell>

# Try to be rigorous about what the voters know at the time of election.

# <codecell>

econ2004 = quarterly_growth.ix[:15].resample('A', 'mean').mean()

# <codecell>

econ2008 = quarterly_growth.ix[15:31].resample('A', 'mean').mean()

# <markdowncell>

# Leave out last quarter 2008 because that's on Bush? Do voters see it that way...?

# <codecell>

econ2012 = quarterly_growth.ix[32:].resample('A', 'mean').mean()

# <markdowncell>

# For flow variables, sum the quarters and get annualized quarter over quarter changes then average.

# <headingcell level=4>

# Get Demographic Data

# <markdowncell>

# Partisan voting index

# <codecell>

pvi = pandas.read_csv("/home/skipper/school/talks/538model/data/partisan_voting.csv")
pvi.set_index("State", inplace=True);
pvi.PVI = pvi.PVI.replace({"EVEN" : "0"})
pvi.PVI = pvi.PVI.str.replace("R\+", "-")
pvi.PVI = pvi.PVI.str.replace("D\+", "")
pvi.PVI = pvi.PVI.astype(float)
pvi.PVI

# <markdowncell>

# Gallup party affiliation (Poll Jan.-Jun. 2012)

# <codecell>

party_affil = pandas.read_csv("/home/skipper/school/talks/538model/"
                              "data/gallup_electorate.csv")
party_affil.Democrat = party_affil.Democrat.str.replace("%", "").astype(float)
party_affil.Republican = party_affil.Republican.str.replace("%", "").astype(float)
party_affil.set_index("State", inplace=True);
party_affil.rename(columns={"Democrat Advantage" : "dem_adv"}, inplace=True);
party_affil["no_party"] = 100 - party_affil.Democrat - party_affil.Republican
party_affil[["dem_adv", "no_party"]]

# <markdowncell>

# Census data

# <codecell>

census_data_2012 = pandas.read_csv("/home/skipper/school/talks/"
                              "538model/data/census_demographics.csv")
def capitalize(s):
    s = s.title()
    s = s.replace("Of", "of")
    return s
census_data_2012["State"] = census_data_2012.state.map(capitalize)
del census_data_2012["state"]
census_data_2012.set_index("State", inplace=True);

# <codecell>

census_data_2000 = pandas.read_csv("/home/skipper/school/talks/"
                                   "538model/data/census_data_2000.csv")
census_data_2000.set_index("State", inplace=True);

# <codecell>

census_data_2005 = (census_data_2000 + census_data_2012) / 2.

# <headingcell level=4>

# Model reversion to the "mean"

# <headingcell level=5>

# A little more data preparation

# <codecell>

changes_2004 = state_polls_election_2004.ix[updated2004].sub(
                    state_polls_oct2_2004)
changes_2004 = changes_2004.dropna()

# <codecell>

changes_2008 = state_polls_election_2008.ix[updated2008].sub(
                    state_polls_oct2_2008)
changes_2008 = changes_2008.dropna()

# <codecell>

changes_2004

# <codecell>

changes_2008

# <codecell>

for name in econ2004.index:
    census_data_2000[name] = econ2004.ix[name]

# <codecell>

for name in econ2008.index:
    census_data_2005[name] = econ2008.ix[name]

# <codecell>

census_data_2000["poll_change"] = changes_2004
census_data_2005["poll_change"] = changes_2008
#changes_2008 = changes_2008.join(census_data_2005)

# <codecell>

#years = pandas.DataFrame([2004]*len(changes_2004), columns=["Year"], index=changes_2004.index)

# <codecell>

#years["poll_change"] = changes_2004
#changes_2004 = years

# <codecell>

#years = pandas.DataFrame([2008]*len(changes_2008), columns=["Year"], index=changes_2008.index)
#years["poll_change"] = changes_2008
#changes_2008 = years

# <codecell>

#changes_2004

# <codecell>

#changes_2004 = changes_2004.join(census_data_2000, how="left")
#changes_2008 = changes_2008.join(census_data_2000, how="left")

# <codecell>

census_data_2000["year"] = 2004
census_data_2005["year"] = 2008

# <codecell>

changes = pandas.concat((census_data_2000.reset_index(), census_data_2005.reset_index()))

# <codecell>

changes.reset_index(drop=True, inplace=True);

# <codecell>

changes = changes.dropna() # don't have polls for all the states

# <codecell>

predict = census_data_2012.reset_index()

# <codecell>

predict["year"] = 2012

# <markdowncell>

# Add in Partisan information

# <codecell>

changes = changes.merge(pvi.reset_index(), on="State")
predict = predict.merge(pvi.reset_index(), on="State")

# <markdowncell>

# Add in Party affiliation information

# <codecell>

changes = changes.merge(party_affil[["dem_adv", "no_party"]].reset_index(), on="State")
predict = predict.merge(party_affil[["dem_adv", "no_party"]].reset_index(), on="State")

# <headingcell level=4>

# Do the K-means clustering for similar states

# <codecell>

from scipy.cluster import vq
from sklearn import cluster

# <codecell>

clstr_dta = predict[["per_black", "per_hisp", "per_white", "educ_coll", "pop_density", "per_older", "PVI", "dem_adv"]].values

# <codecell>

clstr_dta = vq.whiten(clstr_dta) # might want to play with this to emphasize dimensions?

# <codecell>

kmeans = cluster.KMeans(n_clusters=7, n_init=100)
kmeans.fit(clstr_dta)
values = kmeans.cluster_centers_
labels = kmeans.labels_

# <codecell>

predict["kmeans_groups"] = labels

# <codecell>

for key, grp in predict.groupby("kmeans_groups"): print key, grp.State.tolist()

# <codecell>

changes = changes.merge(predict[["kmeans_groups", "State"]], on="State")

# <markdowncell>

# Drop D.C. because it's not in the training data.

# <codecell>

predict.set_index(["State", "year"], inplace=True);

# <codecell>

predict = predict.drop(("District of Columbia", 2012))

# <headingcell level=4>

# Let's explore some hypotheses

# <codecell>

changes.set_index(["State", "year"], inplace=True);

# <codecell>

from statsmodels.formula.api import ols

# <codecell>

changes

# <codecell>

changes[["dem_adv", "PVI"]].corr()

# <codecell>

formula = ("poll_change ~ C(kmeans_groups) + per_older*per_white + "
           "per_hisp + no_party*np.log(median_income) + PVI")
mod = ols(formula, data=changes).fit()
print mod.summary()

# <codecell>

hyp = ", ".join(mod.model.exog_names[:5])

# <codecell>

print hyp

# <codecell>

print mod.f_test(hyp)

# <codecell>

predicted2012 = pandas.read_csv("/home/skipper/school/talks/538model/2012-predicted.csv")
predicted2012["year"] = 2012
predicted2012 = predicted2012.set_index(["State", "year"])["poll"]

# <codecell>

predicted_change = pandas.Series(mod.predict(predict), index=predict.index)

# <codecell>

predicted_change

# <codecell>

results = predicted2012 + predicted_change
results

# <codecell>

electoral_votes = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/electoral_votes.csv")
electoral_votes.sort("State", inplace=True).reset_index(drop=True, inplace=True);
red_states = ["Alabama", "Alaska", "Arkansas", "Idaho", "Kentucky", "Louisiana",
              "Oklahoma", "Wyoming"]
blue_states = ["Delaware"]#, "District of Columbia"]
results.name = "Poll"
results = results.reset_index()
results = results.merge(electoral_votes, on="State", how="left").set_index("State")
results["obama"] = 0
results["romney"] = 0
results.ix[results["Poll"] > 0, ["obama"]] = 1
results.ix[results["Poll"] < 0, ["romney"]] = 1
results.ix[red_states, ["romney"]] = 1
results.ix[blue_states, ["obama"]] = 1

# <codecell>

print results["Votes"].mul(results["obama"]).sum() + 3
print results["Votes"].mul(results["romney"]).sum() 

# <headingcell level=3>

# CCPR Plots

# <markdowncell>

# Component-Component plus residual plots. Partial residual plots attempt to show the relationship between a given independent variable and the response variable given that other independent variables are also in the model.

# <codecell>

from statsmodels.graphics.regressionplots import plot_ccpr_ax
fig, ax = plt.subplots(figsize=(12,8))
fig = plot_ccpr_ax(mod, 11, ax=ax)
ax = fig.axes[0]
ax.set_title("log(median_income)*B_11 + Resid vs log(median_income)");

# <codecell>

from statsmodels.graphics.regressionplots import plot_ccpr_ax
fig, ax = plt.subplots(figsize=(12,8))
fig = plot_ccpr_ax(mod, 9, ax=ax)
ax = fig.axes[0]
ax.set_title("per_hisp*B_9 + resid vs per_hisp");

# <codecell>

X = mod.model.data.orig_exog

# <codecell>

X[X.columns[:6]]

# <codecell>

X[X.columns[6:]]

# <codecell>

false_disc = mod.outlier_test("fdr_bh")
false_disc.sort("unadj_p", inplace=True)

# <codecell>

bonf = mod.outlier_test("sidak")
bonf.sort("unadj_p", inplace=True)

# <codecell>

infl = mod.get_influence()
table = infl.summary_frame()

# <codecell>

for stat in table.columns:
    print stat

# <markdowncell>

# Measure the influence of points on prediction
# 
# $$\text{DFFITS}=\frac{\hat{y}-\hat{y}_{i}}{s_i\sqrt{h_{ii}} }$$
# 
# points greater than
# 
# $$2\left\(\frac{p}{\text{nobs}} \right\)^{1/2}$$
# 
# might be cause for concern

# <codecell>

print 2*np.sqrt(mod.df_model/mod.nobs)

# <codecell>

dffits = np.abs(table['dffits'].copy())
dffits.sort()
dffits[::-1][:15]

# <markdowncell>

# Indicate influential observations, where you might want more data. 
# 
# Overall fit change with deleted observation.
# 
# $$\text{Cook's D}=\frac{e_i^2}{p\text{MSE}\frac{h_{ii}}{(1-h_{ii})^2}}$$

# <codecell>

print 4/mod.nobs

# <codecell>

cooks_d = table["cooks_d"].copy()
cooks_d.sort()
print cooks_d[::-1][:15]

# <codecell>

student_resid = np.abs(table.student_resid.copy())
student_resid.sort()
student_resid[::-1][:15]

