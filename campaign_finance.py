"""
Aggregate contributions
"""
import zipfile

import pandas

fin = zipfile.ZipFile("/home/skipper/school/seaboldgit/talks/pydata/data/fec_obama_itemized_indiv.zip")
fin = fin.open(fin.filelist[0].filename)

obama = pandas.read_csv(fin, skiprows=7)
obama = obama[["State", "Amount", "City", "Zip"]]
obama.Amount = obama.Amount.str.replace("\$", "").astype(float)
obama.State = obama.State.replace({"PE" : "PA"}) # typo

#AE, AA, AP, ZZ are armed forces or canada, etc.
#AB is Canada
state_contrib = obama.groupby("State")["Amount"].sum()
state_contrib = state_contrib.drop(["AE", "AA", "AP", "AB",
                    "AS", # American Samoa
                    "BC",
                    "BR",
                    "FM", # Micronesia
                    "GU", #Guam
                    "MP", #Marianas Islands
                    "NO",
                    "ON",
                    "PR",
                    "QU",
                    "SA",
                    "ZZ",
                    "VI",
                    ])

state_contrib.to_csv("data/obama_indiv_state.csv")

fin = zipfile.ZipFile("/home/skipper/school/seaboldgit/talks/pydata/data/fec_romney_itemized_indiv.zip")
fin = fin.open(fin.filelist[0].filename)

romney = pandas.read_csv(fin, skiprows=7)
romney = romney[["State", "Amount", "City", "Zip"]]
romney.Amount = romney.Amount.str.replace("\$", "").astype(float)
romney.State = romney.State.replace({"TE" : "PN", "GE" : "GA",
                                "PN" : "TN", "HA" : "HI"}) # typo or outdated

state_contrib = romney.groupby("State")["Amount"].sum()
state_contrib = state_contrib.drop(["AE", "AA", "AP",
                    "AS", # American Samoa
                    "GU", #Guam
                    "MP", #Marianas Islands
                    "PR",
                    "VI",
                    "XX",
                    "FF"
                    ])

state_contrib.to_csv("data/romney_indiv_state.csv")
