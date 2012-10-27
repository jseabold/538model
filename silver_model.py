# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# GitHub link for the talk. You can clone the data and play with it yourself. Please submit any improvements as pull requests
# 
# [https://github.com/jseabold/538model](https://github.com/jseabold/538model)

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

# <codecell>

today = datetime.datetime(2012, 10, 2)

# <headingcell level=2>

# Outline

# <markdowncell>

# Methodology was obtained from the old [538 Blog](http://www.fivethirtyeight.com/2008/03/frequently-asked-questions-last-revised.html) with updates at the [new site hosted by the New York Times](http://fivethirtyeight.blogs.nytimes.com/methodology/)

# <markdowncell>

# 1. Polling Average: Aggregate polling data, and weight it according to our reliability scores.
# 
# 2. Trend Adjustment: Adjust the polling data for current trends.
# 
# 3. Regression: Analyze demographic data in each state by means of regression analysis.
# 
# 4. Snapshot: Combine the polling data with the regression analysis to produce an electoral snapshot. This is our estimate of what would happen if the election were held today.
# 
# 5. Projection: Translate the snapshot into a projection of what will happen in November, by allocating out undecided voters and applying a discount to current polling leads based on historical trends.
# 
# 6. Simulation: Simulate our results 10,000 times based on the results of the projection to account for the uncertainty in our estimates. The end result is a robust probabilistic assessment of what will happen in each state as well as in the nation as a whole. 

# <headingcell level=2>

# Get the Data

# <headingcell level=3>

# Consensus forecast of GDP growth over the next two economic quarters <br />(Median of WSJ's monthly forecasting panel)

# <markdowncell>

# The process for creating an economic index for the 538 model is described [here](http://fivethirtyeight.blogs.nytimes.com/2012/07/05/measuring-the-effect-of-the-economy-on-elections/#more-31732).

# <rawcell>

# Obtained from WSJ.com on 10/2/12

# <codecell>

forecasts = pandas.read_table("/home/skipper/school/seaboldgit/"
                              "talks/pydata/data/wsj_forecast.csv", skiprows=2)

# <codecell>

forecasts

# <codecell>

forecasts.rename(columns={"Q3 2012" : "gdp_q3_2012", 
                          "Q4 2012" : "gdp_q4_2012"}, inplace=True)

# <codecell>

forecasts

# <rawcell>

# Pandas methods are NaN aware, so we can just get the median.

# <codecell>

median_forecast = forecasts[['gdp_q3_2012', 'gdp_q4_2012']].median()

# <codecell>

median_forecast

# <headingcell level=3>

# Economics State Variables from FRED

# <markdowncell>

# Job Growth (Nonfarm-payrolls) **PAYEMS** <br />
# Personal Income **PI** <br />
# Industrial production **INDPRO** <br />
# Consumption **PCEC96** <br />
# Inflation **CPIAUCSL** <br />

# <codecell>

from pandas.io.data import DataReader

# <codecell>

series = dict(jobs = "PAYEMS",
              income = "PI",
              prod = "INDPRO",
              cons = "PCEC96",
              prices = "CPIAUCSL")

# <codecell>

#indicators = []
#for variable in series:
#    data = DataReader(series[variable], "fred", start="2010-1-1")
#    # renaming not necessary in master
#    data.rename(columns={"VALUE" : variable}, inplace=True)
#    indicators.append(data)

# <codecell>

#indicators = pandas.concat(indicators, axis=1)

# <codecell>

#indicators

# <headingcell level=3>

# Polling Data

# <markdowncell>

# I used Python to scrape the [Real Clear Politics](realclearpolitics.com) website and download data for the 2004 and 2008 elections. The scraping scripts are available in the github repository for this talk. State by state historical data for the 2004 and 2008 Presidential elections was obtained from [electoral-vote.com](www.electorical-vote.com).

# <headingcell level=2>

# Polling Average

# <markdowncell>

# Details can be found at the 538 blog [here](http://www.fivethirtyeight.com/2008/03/pollster-ratings-updated.html).

# <codecell>

tossup = ["Colorado", "Florida", "Iowa", "New Hampshire", "Nevada", 
          "Ohio", "Virginia", "Wisconsin"]

# <codecell>

national_data2012 = pandas.read_table("/home/skipper/school/seaboldgit/talks/pydata/"
                                      "data/2012_poll_data.csv")

# <codecell>

national_data2012

# <codecell>

national_data2012.rename(columns={"Poll" : "Pollster"}, inplace=True)
national_data2012["obama_spread"] = national_data2012["Obama (D)"] - national_data2012["Romney (R)"]

# <codecell>

national_data2012["State"] = "USA"

# <codecell>

state_data2012 = pandas.read_table("/home/skipper/school/seaboldgit/talks/pydata/data/2012_poll_data_states.csv")
state_data2012

# <codecell>

state_data2012["obama_spread"] = state_data2012["Obama (D)"] - state_data2012["Romney (R)"]

# <codecell>

state_data2012.rename(columns=dict(Poll="Pollster"), inplace=True);

# <codecell>

state_data2012.MoE

# <codecell>

state_data2012.MoE = state_data2012.MoE.replace("--", "nan").astype(float)

# <codecell>

state_data2012

# <codecell>

state_data2012 = state_data2012.set_index(["Pollster", "State", "Date"]).drop("RCP Average", level=0).reset_index()

# <codecell>

state_data2012.head(5)

# <markdowncell>

# Clean up the sample numbers to make it a number.

# <codecell>

state_data2012.Sample

# <codecell>

state_data2012.Sample = state_data2012.Sample.str.replace("\s*([L|R]V)|A", "") # 20 RV
state_data2012.Sample = state_data2012.Sample.str.replace("\s*--", "nan") # --
state_data2012.Sample = state_data2012.Sample.str.replace("^$", "nan")

national_data2012.Sample = national_data2012.Sample.str.replace("\s*([L|R]V)|A", "") # 20 RV
national_data2012.Sample = national_data2012.Sample.str.replace("\s*--", "nan") # --
national_data2012.Sample = national_data2012.Sample.str.replace("^$", "nan")

# <codecell>

state_data2012.Sample.astype(float)

# <codecell>

state_data2012.Sample = state_data2012.Sample.astype(float)
national_data2012.Sample = national_data2012.Sample.astype(float)

# <markdowncell>

# The 2012 data is currently in order of time by state but doesn't have any years.

# <codecell>

#dates2012.get_group(("OH", "NBC News/Marist"))

# <codecell>

state_data2012["start_date"] = ""
state_data2012["end_date"] = ""
dates2012 = state_data2012.groupby(["State", "Pollster"])["Date"]
for _, date in dates2012:
    year = 2012
    # checked by hand, none straddle years
    changes = np.r_[False, np.diff(map(int, [i[0].split('/')[0] for 
                    i in date.str.split(' - ')])) > 0]
    for j, (idx, dt) in enumerate(date.iteritems()):
        dt1, dt2 = dt.split(" - ")
        year -= changes[j]
        # check for ones that haven't polled in a year - soft check
        # could be wrong for some...
        if year == 2012 and (int(dt1.split("/")[0]) > today.month and 
                             int(dt1.split("/")[1]) > today.day):
            year -= 1
        dt1 += "/" + str(year)
        dt2 += "/" + str(year)
        state_data2012.set_value(idx, "start_date", dt1)
        state_data2012.set_value(idx, "end_date", dt2)

# <codecell>

national_data2012["start_date"] = ""
national_data2012["end_date"] = ""
dates2012 = national_data2012.groupby(["Pollster"])["Date"]
for _, date in dates2012:
    year = 2012
    # checked by hand, none straddle years
    changes = np.r_[False, np.diff(map(int, [i[0].split('/')[0] for 
                    i in date.str.split(' - ')])) > 0]
    for j, (idx, dt) in enumerate(date.iteritems()):
        dt1, dt2 = dt.split(" - ")
        year -= changes[j]
        dt1 += "/" + str(year)
        dt2 += "/" + str(year)
        national_data2012.set_value(idx, "start_date", dt1)
        national_data2012.set_value(idx, "end_date", dt2)

# <codecell>

state_data2012.head(10)

# <codecell>

state_data2012.start_date = state_data2012.start_date.apply(pandas.datetools.parse)
state_data2012.end_date = state_data2012.end_date.apply(pandas.datetools.parse)

national_data2012.start_date = national_data2012.start_date.apply(pandas.datetools.parse)
national_data2012.end_date = national_data2012.end_date.apply(pandas.datetools.parse)

# <codecell>

def median_date(row):
    dates = pandas.date_range(row["start_date"], row["end_date"])
    median_idx = int(np.median(range(len(dates)))+.5)
    return dates[median_idx]
    
state_data2012["poll_date"] = [median_date(row) for i, row in state_data2012.iterrows()]
del state_data2012["Date"]
del state_data2012["start_date"]
del state_data2012["end_date"]

national_data2012["poll_date"] = [median_date(row) for i, row in national_data2012.iterrows()]
del national_data2012["Date"]
del national_data2012["start_date"]
del national_data2012["end_date"]

# <codecell>

state_data2012.head(5)

# <codecell>

pollsters = state_data2012.Pollster.unique()
pollsters.sort()

# <codecell>

len(pollsters)

# <codecell>

print pandas.Series(pollsters)

# <headingcell level=3>

# 538 Pollster Ratings

# <codecell>

weights = pandas.read_table("/home/skipper/school/seaboldgit/talks/pydata/data/pollster_weights.csv")

# <codecell>

weights

# <codecell>

weights.mean()

# <markdowncell>

# Clean up the pollster names a bit so we can merge with the weights.

# <codecell>

import pickle
pollster_map = pickle.load(open("/home/skipper/school/seaboldgit/talks/pydata/data/pollster_map.pkl", "rb"))

# <codecell>

state_data2012.Pollster.replace(pollster_map, inplace=True);

# <codecell>

national_data2012.Pollster.replace(pollster_map, inplace=True);

# <markdowncell>

# Inner merge the data with the weights

# <codecell>

state_data2012 = state_data2012.merge(weights, how="inner", on="Pollster")

# <codecell>

state_data2012.head(5)

# <headingcell level=4>

# First, we average each pollster for each state.

# <markdowncell>

# The first adjustment is an exponential decay for recency of the poll. Based on research in prior elections, a weight with a half-life of 30 days since the median date the poll has been in the field is assigned to each poll.

# <codecell>

def exp_decay(days):
    # defensive coding, accepts timedeltas
    days = getattr(days, "days", days)
    return .5 ** (days/30.)

# <codecell>

fig, ax = plt.subplots(figsize=(12,8), subplot_kw={"xlabel" : "Days",
                                                   "ylabel" : "Weight"})
days = np.arange(0, 45)
ax.plot(days, exp_decay(days));
ax.vlines(30, 0, .99, color='r', linewidth=4)
ax.set_ylim(0,1)
ax.set_xlim(0, 45);

# <markdowncell>

# The second adjustment is for the sample size of the poll. Polls with a higher sample size receive a higher weight.

# <markdowncell>

# Binomial sampling error = +/- $50 * \frac{1}{\sqrt{nobs}}$ where the 50 depends on the underlying probability or population preferences, in this case assumed to be 50:50 (another way of calculating Margin of Error)

# <codecell>

def average_error(nobs, p=50.):
    return p*nobs**-.5

# <markdowncell>

# The thinking here is that having 5 polls of 1200 is a lot like having one poll of 6000. However, we downweight older polls by only including the marginal effective sample size. Where the effective sample size is the size of the methodologically perfect poll for which we would be indifferent between it and the one we have with our current total error. Total error is determined as $TE = \text{Average Error} + \text{Long Run Pollster Induced Error}$. See [here](http://www.fivethirtyeight.com/2008/04/pollster-ratings-v30.html) for the detailed calculations of Pollster Induced Error.

# <codecell>

def effective_sample(total_error, p=50.):
    return p**2 * (total_error**-2.)

# <codecell>

state_pollsters = state_data2012.groupby(["State", "Pollster"])

# <codecell>

ppp_az = state_pollsters.get_group(("AZ", "Public Policy Polling (PPP)"))

# <codecell>

var_idx = ["Pollster", "State", "Obama (D)", "Romney (R)", "Sample", "poll_date"]
ppp_az[var_idx]

# <codecell>

ppp_az.sort("poll_date", ascending=False, inplace=True);

# <codecell>

ppp_az["cumulative"] = ppp_az["Sample"].cumsum()
ppp_az["average_error"] = average_error(ppp_az["cumulative"])
ppp_az["total_error"] = ppp_az["PIE"] + ppp_az["average_error"]
ppp_az[var_idx + ["cumulative"]]

# <codecell>

ppp_az["ESS"] = effective_sample(ppp_az["total_error"])
ppp_az["MESS"] = ppp_az["ESS"].diff()
# fill in first one
ppp_az["MESS"].fillna(ppp_az["ESS"].head(1).item(), inplace=True);

# <codecell>

ppp_az[["poll_date", "Sample", "cumulative", "ESS", "MESS"]]

# <markdowncell>

# Now let's do it for every polling firm in every state.

# <codecell>

def calculate_mess(group):
    cumulative = group["Sample"].cumsum()
    ae = average_error(cumulative)
    total_error = ae + group["PIE"]
    ess = effective_sample(total_error)
    mess = ess.diff()
    mess.fillna(ess.head(1).item(), inplace=True)
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    return pandas.concat((ess, mess), axis=1)

#state_data2012["ESS", "MESS"] 
df = state_pollsters.apply(calculate_mess)
df.rename(columns={0 : "ESS", 1 : "MESS"}, inplace=True);

# <codecell>

state_data2012 = state_data2012.join(df)

# <markdowncell>

# Give them the time weight

# <codecell>

td = today - state_data2012["poll_date"].head(1).item()

# <codecell>

state_data2012["poll_date"].head(1).item()

# <codecell>

td

# <codecell>

state_data2012["time_weight"] = (today - state_data2012["poll_date"]).apply(exp_decay)

# <markdowncell>

# Now aggregate all of these. Weight them based on the sample size but also based on the time_weight.

# <codecell>

def weighted_mean(group):
    weights1 = group["time_weight"]
    weights2 = group["MESS"]
    return np.sum(weights1*weights2*group["obama_spread"]/(weights1*weights2).sum())

# <codecell>

state_pollsters = state_data2012.groupby(["State", "Pollster"])
state_polls = state_pollsters.apply(weighted_mean)

# <codecell>

state_polls

# <headingcell level=3>

# 2004 and 2008 Polls

# <codecell>

state_data2004 = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2004-pres-polls.csv")
state_data2004

# <codecell>

state_data2004.head(5)

# <codecell>

state_data2008 = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2008-pres-polls.csv")
state_data2008

# <codecell>

state_data2008.head(5)

# <codecell>

state_data2008.End + " 2008"

# <codecell>

(state_data2008.End + " 2008").apply(pandas.datetools.parse)

# <markdowncell>

# Need to clean some of the dates in this data. Luckily, pandas makes this easy to do.

# <codecell>

state_data2004.Date = state_data2004.Date.str.replace("Nov 00", "Nov 01")
state_data2004.Date = state_data2004.Date.str.replace("Oct 00", "Oct 01")

# <codecell>

state_data2008["poll_date"] = (state_data2008.End + " 2008").apply(pandas.datetools.parse)
state_data2004["poll_date"] = (state_data2004.Date + " 2004").apply(pandas.datetools.parse)

# <codecell>

del state_data2008["End"]
del state_data2008["Start"]
del state_data2004["Date"]

# <codecell>

state_data2008

# <codecell>

state_data2004

# <codecell>

state_groups = state_data2008.groupby("State")

# <codecell>

state_groups.aggregate(dict(Obama=np.mean, McCain=np.mean))

# <markdowncell>

# Means for the entire country (without weighting by population)

# <codecell>

state_groups.aggregate(dict(Obama=np.mean, McCain=np.mean)).mean()

# <codecell>

state_data2004.Pollster.replace(pollster_map, inplace=True)
state_data2008.Pollster.replace(pollster_map, inplace=True);

# <codecell>

state_data2004 = state_data2004.merge(weights, how="inner", on="Pollster")
state_data2008 = state_data2008.merge(weights, how="inner", on="Pollster")

# <codecell>

len(state_data2004.Pollster.unique())

# <codecell>

len(state_data2008.Pollster.unique())

# <codecell>

import datetime

# <codecell>

date2004 = datetime.datetime(2004, 11, 2)
date2004

# <codecell>

(date2004 - state_data2004.poll_date) < datetime.timedelta(21)

# <markdowncell>

# Restrict the samples to the 3 weeks leading up to the election

# <codecell>

state_data2004 = state_data2004.ix[(date2004 - state_data2004.poll_date) <= datetime.timedelta(21)]
state_data2004.reset_index(drop=True, inplace=True)

# <codecell>

date2008 = datetime.datetime(2008, 11, 4)

# <codecell>

state_data2008 = state_data2008.ix[(date2008 - state_data2008.poll_date) <= datetime.timedelta(21)]
state_data2008.reset_index(drop=True, inplace=True)

# <codecell>

state_data2008

# <codecell>

state_data2004["time_weight"] =(date2004 - state_data2004.poll_date).apply(exp_decay)
state_data2008["time_weight"] =(date2008 - state_data2008.poll_date).apply(exp_decay)

# <codecell>

state_data2004[["time_weight", "poll_date"]].head(5)

# <codecell>

def max_date(x):
    return x == x.max()

# <codecell>

state_data2004["newest_poll"] = state_data2004.groupby(("State", "Pollster")).poll_date.transform(max_date)
state_data2008["newest_poll"] = state_data2008.groupby(("State", "Pollster")).poll_date.transform(max_date)

# <headingcell level=3>

# Clustering States by Demographics

# <markdowncell>

# There are notes on trend line adjustment, [here](http://www.fivethirtyeight.com/2008/06/we-know-more-than-we-think-big-change-2.html), [here](http://www.fivethirtyeight.com/2008/06/refinement-to-adjustment-part-i.html), [here](http://www.fivethirtyeight.com/2008/06/refinement-to-adjustment-part-ii.html), [here](http://www.fivethirtyeight.com/2008/06/trendline-now-calculated-from-daily.html), and [here](http://www.fivethirtyeight.com/2008/06/construction-season-over-technical.html). However, to the best of my knowledge, the similar state "nearest neighbor" clustering remains a black box.

# <markdowncell>

# Partican Voting Index data obtained from [Wikipedia](http://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index)

# <codecell>

pvi = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/partisan_voting.csv")

# <codecell>

pvi.set_index("State", inplace=True);
pvi

# <codecell>

pvi.PVI = pvi.PVI.replace({"EVEN" : "0"})
pvi.PVI = pvi.PVI.str.replace("R\+", "-")
pvi.PVI = pvi.PVI.str.replace("D\+", "")
pvi.PVI = pvi.PVI.astype(float)
pvi.PVI

# <markdowncell>

# Party affliation of electorate obtained from [Gallup](http://www.gallup.com/poll/156437/Heavily-Democratic-States-Concentrated-East.aspx#2).

# <codecell>

party_affil = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/gallup_electorate.csv")

# <codecell>

party_affil.Democrat = party_affil.Democrat.str.replace("%", "").astype(float)
party_affil.Republican = party_affil.Republican.str.replace("%", "").astype(float)
party_affil.set_index("State", inplace=True);
party_affil.rename(columns={"Democrat Advantage" : "dem_adv"}, inplace=True);
party_affil["no_party"] = 100 - party_affil.Democrat - party_affil.Republican

# <codecell>

party_affil

# <codecell>

census_data = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/census_demographics.csv")

# <codecell>

def capitalize(s):
    s = s.title()
    s = s.replace("Of", "of")
    return s

# <codecell>

census_data["State"] = census_data.state.map(capitalize)
del census_data["state"]

# <codecell>

census_data.set_index("State", inplace=True)

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

# <markdowncell>

# Campaign Contributions from FEC.

# <codecell>

obama_give = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/obama_indiv_state.csv", 
                             header=None, names=["State", "obama_give"])
romney_give = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/romney_indiv_state.csv",
                             header=None, names=["State", "romney_give"])

# <codecell>

obama_give.State.replace(states_abbrev_dict, inplace=True);
romney_give.State.replace(states_abbrev_dict, inplace=True);

# <codecell>

obama_give.set_index("State", inplace=True)
romney_give.set_index("State", inplace=True);

# <codecell>

demo_data = census_data.join(party_affil[["dem_adv", "no_party"]]).join(pvi)

# <codecell>

demo_data = demo_data.join(obama_give).join(romney_give)

# <codecell>

giving = demo_data[["obama_give", "romney_give"]].div(demo_data[["vote_pop", "older_pop"]].sum(1), axis=0)
giving

# <codecell>

demo_data[["obama_give", "romney_give"]] = giving

# <codecell>

from scipy import cluster as sp_cluster
from sklearn import cluster, neighbors

# <codecell>

clean_data = sp_cluster.vq.whiten(demo_data.values)

# <codecell>

clean_data.var(axis=0)

# <codecell>

KNN = neighbors.NearestNeighbors(n_neighbors=7)
KNN.fit(clean_data)
KNN.kneighbors(clean_data[0], return_distance=True)

# <codecell>

idx = _[1]
demo_data.index[0], demo_data.index[idx]

# <codecell>

nearest_neighbor = {}
for i, state in enumerate(demo_data.index):
    neighborhood = KNN.kneighbors(clean_data[i], return_distance=True)
    nearest_neighbor.update({state : (demo_data.index[neighborhood[1]],
                                     neighborhood[0])})

# <codecell>

nearest_neighbor

# <codecell>

k_means = cluster.KMeans(n_clusters=5, n_init=50)
k_means.fit(clean_data)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

# <codecell>

clusters = sp_cluster.vq.kmeans(clean_data, 5)[0]

# <codecell>

def choose_group(data, clusters):
    """
    Return the index of the cluster to which the rows in data
    are "closest" (in the sense of the L2-norm)
    """
    data = data[:,None] # add an axis for broadcasting
    distances = data - clusters
    groups = []
    for row in distances:
        dists = map(np.linalg.norm, row)
        groups.append(np.argmin(dists))
    return groups

# <codecell>

groups = choose_group(clean_data, clusters)

# <codecell>

np.array(groups)

# <markdowncell>

# Or use a one-liner

# <codecell>

groups = [np.argmin(map(np.linalg.norm, (clean_data[:,None] - clusters)[i])) for i in range(51)]

# <codecell>

demo_data["kmeans_group"] = groups
demo_data["kmeans_labels"] = labels

# <codecell>

for _, group in demo_data.groupby("kmeans_group"):
    group = group.index
    group.values.sort()
    print group.values

# <codecell>

labels

# <codecell>

demo_data["kmeans_labels"] = labels
for _, group in demo_data.groupby("kmeans_labels"):
    group = group.index.copy()
    group.values.sort()
    print group.values

# <codecell>

demo_data = demo_data.reset_index()

# <codecell>

state_data2012.State.replace(states_abbrev_dict, inplace=True);

# <codecell>

state_data2012 = state_data2012.merge(demo_data[["State", "kmeans_labels"]], on="State")

# <codecell>

kmeans_groups = state_data2012.groupby("kmeans_labels")

# <codecell>

group = kmeans_groups.get_group(kmeans_groups.groups.keys()[2])

# <codecell>

group.State.unique()

# <codecell>

def edit_tick_label(tick_val, tick_pos):
    if tick_val  < 0:
        text = str(int(tick_val)).replace("-", "Romney+")
    else:
        text = "Obama+"+str(int(tick_val))
    return text

# <codecell>

from pandas import lib
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(figsize=(12,8))

data = group[["poll_date", "obama_spread"]]
data = pandas.concat((data, national_data2012[["poll_date", "obama_spread"]]))
    
data.sort("poll_date", inplace=True)
dates = pandas.DatetimeIndex(data.poll_date).asi8

loess_res = sm.nonparametric.lowess(data.obama_spread.values, dates, 
                                    frac=.2, it=3)

dates_x = lib.ints_to_pydatetime(dates)
axes.scatter(dates_x, data["obama_spread"])
axes.plot(dates_x, loess_res[:,1], color='r')
axes.yaxis.get_major_locator().set_params(nbins=12)
axes.yaxis.set_major_formatter(FuncFormatter(edit_tick_label))
axes.grid(False, axis='x')
axes.hlines(0, dates_x[0], dates_x[-1], color='black', lw=3)
axes.margins(0, .05)

# <codecell>

loess_res[-7:,1].mean()

# <codecell>

from pandas import lib
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(figsize=(12,8))

national_data2012.sort("poll_date", inplace=True)
dates = pandas.DatetimeIndex(national_data2012.poll_date).asi8

loess_res = sm.nonparametric.lowess(national_data2012.obama_spread.values, dates, 
                                    frac=.075, it=3)

dates_x = lib.ints_to_pydatetime(dates)
axes.scatter(dates_x, national_data2012["obama_spread"])
axes.plot(dates_x, loess_res[:,1], color='r')
axes.yaxis.get_major_locator().set_params(nbins=12)
axes.yaxis.set_major_formatter(FuncFormatter(edit_tick_label))
axes.grid(False, axis='x')
axes.hlines(0, dates_x[0], dates_x[-1], color='black', lw=3)
axes.margins(0, .05)

# <codecell>

trends = []
for i, group in kmeans_groups:
    data = group[["poll_date", "obama_spread"]]
    data = pandas.concat((data, national_data2012[["poll_date", "obama_spread"]]))
    
    data.sort("poll_date", inplace=True)
    dates = pandas.DatetimeIndex(data.poll_date).asi8

    loess_res = sm.nonparametric.lowess(data.obama_spread.values, dates, 
                                    frac=.1, it=3)
    states = group.State.unique()
    for state in states:
        trends.append([state, loess_res[-7:,1].mean()])

# <codecell>

trends

# <headingcell level=4>

# Adjust for sensitivity to time-trends

# <markdowncell>

# $$\text{Margin}=X_i+Z_t+\epsilon$$
# 
# where $S_i$ are Pollster:State dummies. In a state with a time-dependent trend, you might write
# 
# $$\text{Margin}=X_i+m*Z_t$$
# 
# where $m$ is a multiplier representing uncertainty in the time-trend parameter. Solving for $m$ gives
# 
# $$m=\text{Margin}-\frac{X_i}{Z_t}$$

# <codecell>

from statsmodels.formula.api import ols, wls

# <codecell>

#pollster_state_dummy = state_data2012.groupby(["Pollster", "State"])["obama_spread"].mean()
#daily_dummy = state_data2012.groupby(["poll_date"])["obama_spread"].mean()

# <codecell>

state_data2012["pollster_state"] = state_data2012["Pollster"] + "-" + state_data2012["State"]

# <markdowncell>

# There's actually a bug in pandas when you merge on datetimes. In order to avoid it, we need to sort our data now and once again after we merge on dates.

# <codecell>

state_data2012.sort(columns=["pollster_state", "poll_date"], inplace=True);

# <codecell>

dummy_model = ols("obama_spread ~ C(pollster_state) + C(poll_date)", data=state_data2012).fit()

# <markdowncell>

# The base case is American Research Group-Colorado

# <codecell>

state_data2012.irow(0)

# <codecell>

pollster_state = state_data2012["pollster_state"].unique()
pollster_state.sort()
pollster_state_params = dummy_model.params[1:len(pollster_state)] + dummy_model.params[0]
intercept = dummy_model.params[0]
X = pandas.DataFrame(zip(pollster_state, np.r_[intercept, pollster_state_params]), 
                     columns=["pollster_state", "X"])

# <codecell>

dates = state_data2012.poll_date.unique()
dates.sort()
dates_params = intercept + dummy_model.params[-len(dates):]
Z = pandas.DataFrame(zip(dates, dates_params), columns=["poll_date", "Z"])

# <markdowncell>

# Drop the ones less than 1.

# <codecell>

Z = Z.ix[np.abs(Z.Z) > 1]

# <codecell>

state_data2012 = state_data2012.merge(X, on="pollster_state", sort=False)
state_data2012 = state_data2012.merge(Z, on="poll_date", sort=False)

# <codecell>

state_data2012.sort(columns=["pollster_state", "poll_date"], inplace=True);

# <codecell>

state_data2012["m"] = state_data2012["obama_spread"].sub(state_data2012["X"].div(state_data2012["Z"]))

# <codecell>

#m_dataframe.ix[m_dataframe.pollster_state == "American Research Group-New Hampshire"].values

# <codecell>

m_dataframe = state_data2012[["State", "m", "poll_date", "Pollster", "pollster_state"]]

# <codecell>

m_dataframe["m"].describe()

# <codecell>

m_size = m_dataframe.groupby("pollster_state").size()

# <codecell>

m_size

# <codecell>

drop_idx = m_size.ix[m_size == 1]

# <codecell>

m_dataframe = m_dataframe.set_index(["pollster_state", "poll_date"])

# <codecell>

m_dataframe.xs("American Research Group-New Hampshire", level=0)

# <codecell>

m_dataframe = m_dataframe.drop(drop_idx.index, level=0).reset_index()

# <codecell>

m_dataframe

# <codecell>

m_regression_data = m_dataframe.merge(demo_data, on="State")

# <codecell>

m_regression_data

# <codecell>

m_regression_data[["PVI", "per_black", "per_hisp", "older_pop", "average_income", 
                   "romney_give", "obama_give", "educ_coll", "educ_hs"]].corr()

# <codecell>

(today - m_regression_data["poll_date"].astype('O'))

# <codecell>

time_weights = (today - m_regression_data["poll_date"].astype('O')).apply(exp_decay)

# <codecell>

m_model = wls("m ~ PVI + per_hisp + per_black + average_income + educ_coll", data=m_regression_data, weights=time_weights).fit()
m_model.summary()

# <codecell>

state_resid = pandas.DataFrame(zip(m_model.resid, m_regression_data.State), 
                               columns=["resid", "State"])

# <codecell>

state_resid_group = state_resid.groupby("State")

# <codecell>

fig, axes = plt.subplots(figsize=(12,8), subplot_kw={"ylabel" : "Residual",
                                                     "xlabel" : "State"})
i = 0
for state, group in state_resid_group:
    x = [i] * len(group)
    axes.scatter(x, group["resid"], s=91)
    i += 1
states = m_regression_data.State.unique()
states.sort()
#axes.xaxis.get_major_locator().set_params(nbins=len(states))
axes.margins(.05, .05)
axes.xaxis.set_ticks(range(31))
axes.xaxis.set_ticklabels(states);
for label in axes.xaxis.get_ticklabels():
    label.set_rotation(90)
    label.set_fontsize('large')

# <codecell>

demo_data = demo_data.drop(demo_data.index[demo_data['State'] == 'District of Columbia'])
demo_data.reset_index(drop=True, inplace=True);

# <codecell>

exog = demo_data[["PVI", "per_hisp", "per_black", "average_income", "educ_coll"]]
exog["const"] = 1

# <codecell>

state_m = m_model.predict(exog)

# <codecell>

state_m

# <codecell>

unit_m = (state_m - state_m.min())/(state_m.max() - state_m.min())

# <codecell>

unit_m *= 2

# <codecell>

m_correction = zip(demo_data.State, unit_m)

# <codecell>

fig, axes = plt.subplots(figsize=(12,8), subplot_kw={"ylabel" : "Time Uncertainty",
                                                     "xlabel" : "State"})

axes.scatter(range(len(unit_m)), unit_m, s=91)

axes.margins(.05, .05)
axes.xaxis.set_ticks(range(len(unit_m)))
axes.xaxis.set_ticklabels(demo_data.State);
for label in axes.xaxis.get_ticklabels():
    label.set_rotation(90)
    label.set_fontsize('large')

# <codecell>

m_correction

# <codecell>

trends = pandas.DataFrame(trends, columns=["State", "trend"])
m_correction = pandas.DataFrame(m_correction, columns=["State", "m_correction"])

# <codecell>

trends = trends.merge(m_correction, on="State")

# <codecell>

trends.set_index("State", inplace=True)

# <codecell>

trends = trends.product(axis=1)

# <headingcell level=3>

# Snapshot: Combine Trend Estimates and State Polls

# <codecell>

state_polls.name = "poll"
state_polls

# <codecell>

state_polls = state_polls.reset_index()
state_polls.State = state_polls.State.replace(states_abbrev_dict)

# <codecell>

trends.name = "poll"
trends

# <codecell>

trends = trends.reset_index()
trends["Pollster"] = "National"

# <codecell>

trends

# <codecell>

polls = pandas.concat((state_polls, trends))

# <codecell>

weights

# <codecell>

natl_weight = pandas.DataFrame([["National", weights.Weight.mean(), weights.PIE.mean()]],
                                columns=["Pollster", "Weight", "PIE"])
weights = pandas.concat((weights, natl_weight)).reset_index(drop=True)

# <codecell>

polls = polls.merge(weights, on="Pollster", how="left")

# <codecell>

polls = polls.sort("State")

# <codecell>

def weighted_mean(group):
    return (group["poll"] * group["Weight"] / group["Weight"].sum()).sum()

# <codecell>

group

# <codecell>

results = polls.groupby("State").aggregate(weighted_mean)["poll"]

# <codecell>

results

# <codecell>

results = results.reset_index()
results["obama"] = 0
results["romney"] = 0
results.ix[results["poll"] > 0, ["obama"]] = 1
results.ix[results["poll"] < 0, ["romney"]] = 1

# <codecell>

results[["State", "poll"]].to_csv("/home/skipper/school/talks/538model/2012-predicted.csv", index=False)

# <codecell>

electoral_votes = pandas.read_csv("/home/skipper/school/seaboldgit/talks/pydata/data/electoral_votes.csv")

# <codecell>

electoral_votes.sort("State", inplace=True).reset_index(drop=True, inplace=True)

# <codecell>

results = electoral_votes.merge(results, on="State", how="left")

# <codecell>

results = results.set_index("State")
red_states = ["Alabama", "Alaska", "Arkansas", "Idaho", "Kentucky", "Louisiana",
              "Oklahoma", "Wyoming"]
blue_states = ["Delaware", "District of Columbia"]
results.ix[red_states, ["romney"]] = 1
results.ix[red_states, ["obama"]] = 0
results.ix[blue_states, ["obama"]] = 1
results.ix[blue_states, ["romney"]] = 0

# <codecell>

results

# <codecell>

results["Votes"].mul(results["obama"]).sum()

# <codecell>

results["Votes"].mul(results["romney"]).sum()

# <codecell>

results

# <markdowncell>

# TODO:

# <markdowncell>

# Divide undecided voters probabilistically.

# <markdowncell>

# Do historical adjustments based on how polls changed in the past conditional on "election environment"

# <markdowncell>

# "Error analysis"

