"""
Census data downloaded from quickfacts.census.gov/qfd/download_data.html
"""
import pandas

# this includes counties, state, and aggregate
# comma delimited
full_data_url = "http://quickfacts.census.gov/qfd/download/DataSet.txt"
# variable info is given here
data_info = "http://quickfacts.census.gov/qfd/download/DataDict.txt"
# these are the mappings that we want - fixed width file
fips_names = "http://quickfacts.census.gov/qfd/download/FIPS_CountyName.txt"


full_data = pandas.read_csv(full_data_url)
fips_names = pandas.read_fwf(fips_names, [(0,5),(6,-1)], header=None,
                             names=["FIPS", "name"])
#NOTE: this is fixed width and I'm not counting the stupid columns so
#we're going to do it programmatically
# note that one of the column headers is centered while the others aren't
from urllib2 import urlopen
import re
headers = urlopen(data_info).readline()
match = re.search("^(\w+\W)(\s{2}\s+\w+\s+\s{2})(\s{2}\w+\s+)"
                  "(\w+\s+)(\w+\s+)(\w+)", headers)
cols = [(headers.index(var_name), headers.index(var_name)+len(var_name)) for
        var_name in match.groups()]
var_info = pandas.read_fwf(data_info, cols)

# convert numbers to state/county names
fips_mapping = {}
for _, (fips, name) in fips_names.iterrows():
    fips_mapping.update({str(fips) : name})

fips_names = full_data.FIPS.astype(str).replace(fips_mapping)
del full_data['FIPS']
full_data['FIPS'] = fips_names

# just keep the states
states = full_data.FIPS.ix[~full_data.FIPS.str.contains(",")]
states = states.ix[~(states == "UNITED STATES")]
assert len(states) == 51
idx = states.index
full_data_states = full_data.ix[idx]

# Total Pop, 1
# % Under 18, 6
# Over 65, 7
# % females, 8
# % Black, 10
# % Native American, 11
# % Hispanic, 15
# % White, Non-Hispanic, 16
# % high school grad, 20
# % bachelor's degree, 21
# per capita income, 30
# median household income, 31
# pop per sq mile, 51
rows = [1,6,7,8,10,11,15,16,20,21,30,31,51]
var_info = var_info.ix[rows][["Data_Item","Item_Description"]]
full_data_states = full_data_states.filter(var_info.Data_Item.tolist() +
                                            ["FIPS"])

tot_pop = full_data["PST045211"]
per_18 = full_data["AGE295211"]/100. # under 18
per_65 = full_data["AGE775211"]/100. # over 65
older_pop = per_65*tot_pop
vote_pop = tot_pop - per_18*tot_pop - older_pop
full_data_states["vote_pop"] = vote_pop
full_data_states["older_pop"] = older_pop
del full_data_states["PST045211"]
del full_data_states["AGE295211"]
del full_data_states["AGE775211"]
del full_data_states["SEX255211"] # % females - not enough variation
del full_data_states["RHI325211"]
full_data_states["per_older"] = older_pop / tot_pop
full_data_states["per_vote"] = vote_pop / tot_pop

full_data_states.rename(columns={
                        "INC110210" : "median_income",
                        "INC910210" : "average_income",
                        "POP060210" : "pop_density",
                        "EDU635210" : "educ_hs",
                        "EDU685210" : "educ_coll",
                        "RHI825211" : "per_white",
                        "RHI725211" : "per_hisp", # not mutually excl.
                        "FIPS"      : "state",
                        "RHI225211" : "per_black",
                                }, inplace=True)
full_data_states.set_index("state", inplace=True)

full_data_states.to_csv("/home/skipper/school/seaboldgit/talks/pydata/"
                        "data/census_demographics.csv")
