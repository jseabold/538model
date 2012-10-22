"""
Download all the Presidential Poll Data from Real Clear Politics and
put it in a DataFrame then put a bird on it.

Usage:

    data = download_president_polls()

Data will probably need some ex-post cleaning to be useful.
"""

raise Exception("This script no longer works as RCP blocks directory access to their servers for some reason. Needs to be updated.")

import urllib2
import re
from urllib import urlencode
from time import sleep

from lxml import html
import pandas

GLOBAL_NUMBER_OF_URL_TRIES = 0

states = {
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

def get_xml_from_url(url, data={}):
    data = urlencode(data)
    request = urllib2.Request(url, data=data)
    opener = urllib2.build_opener()
    try:
        response = opener.open(request)
    except urllib2.HTTPError, error:
        return error
    xml_tree = html.parse(response)
    return xml_tree

def split_row(row):
    return [i.text_content() for i in row.getchildren()]

def get_table(table):
    table_data = []
    for i, child in enumerate(table.iterchildren()):
        if not (i % 2):
            date = child.text_content()
        else: # iterate through the sub-table
            if not table_data: # then get the headers
                header = split_row(child[0])
                header[0] = re.sub("\xa0\xa0\(.*\)", "", header[0]).strip()
                table_data.append(header + ["Date"])
            for row in child.getchildren()[1:]:
                row = split_row(row)
                table_data.append(row + [date])
    return pandas.DataFrame(table_data[1:], columns=table_data[0])


def download_latest_state_polls():
    url = "http://www.realclearpolitics.com/epolls/latest_polls/president/"
    table_data = []
    #table should go date, table, date, table, until end, then follow next
    #linke
    xml_tree = get_xml_from_url(url)
    i = 1
    # right now there are 3 table classes to get. don't know if this is
    # always the case, so code defensively

    table = xml_tree.xpath('//div[@id="table-%d"]' % i)
    table_frame = get_table(table[0])
    while table:
        i += 1
        table = xml_tree.xpath('//div[@id="table-%d"]' % i)
        if table:
            table_frame = pandas.concat((table_frame, get_table(table[0])))

    return table_frame

def politely_open_url(url):
    """
    This 403s a bit while doing the server load balancing, so politely knock
    again.
    """
    global GLOBAL_NUMBER_OF_URL_TRIES
    try:
        response = urllib2.urlopen(url)
        GLOBAL_NUMBER_OF_URL_TRIES = 0
        return response
    except urllib2.HTTPError, error:
        if GLOBAL_NUMBER_OF_URL_TRIES > 100:
            print "More than 100 tries of %s failed" % url
            return error
        elif error.msg == 'Forbidden':
            GLOBAL_NUMBER_OF_URL_TRIES += 1
            sleep(1)
            return politely_open_url(url)
        else:
            return error


def download_state_polls():
    """
    Use this to download all state polls. Unforunately, there's no years on a lot of the
    data. But you should be able to cross-reference with latest.
    """
    # need to walk this directory for the states plus DC
    url = "http://www.realclearpolitics.com/epolls/2012/president/"

    # inside each directory get the one called state_romney_vs_obama*.html
    state_xx = [key.lower() for key in states.keys()]

    table_data = []
    for state in states:
        url_link = url + state.lower() + '/'
        response = politely_open_url(url_link)
        xml_tree = html.parse(response)
        state_links = xml_tree.findall('//a')
        for link in state_links:
            link.make_links_absolute()
            link_url = link.get('href')
            # i don't think we have to worry about there being more than one
            if re.match("http://.+_romney_vs_obama-.+\.html", link_url):
                print "Trying to download %s" % link_url
                xml_tree = get_xml_from_url(link_url)
                try:
                    # some states like Alaska and Alabama don't have any polls?
                    table = xml_tree.xpath(
                            '//div[@id="polling-data-full"]//table')[0]
                    print "Downloaded %s" % link_url
                except:
                    continue
                # this should work for states too
                state_table = get_national_tables(table)
                state_table["State"] = state
                table_data.append(state_table)
        # chill out for a second. server seems finnicky
        sleep(1)
    table_2012 = pandas.concat(table_data)
    return table_2012

def get_national_tables(table):
    table_data = []
    for row in table.iterchildren():
        row = split_row(row)
        table_data.append(row)
    return pandas.DataFrame(table_data[1:], columns=table_data[0])


def download_national_polls():
    #NOTE: the 2012 data is likely to update daily

    # you can browse around from here, sometimes I get forbidden, sometimes not
    # seems to depend which page I come from AFIACT. Refreshing also works.
    # I think it's bouncing me around to different servers.
    #http://www.realclearpolitics.com/epolls/
    # bush vs gore
    # can't find anything, just an electoral college map

    # bush vs kerry
    url = "http://www.realclearpolitics.com/epolls/2004/president/us/general_election_bush_vs_kerry-939.html"
    xml_tree = get_xml_from_url(url)
    table = xml_tree.xpath('//div[@id="polling-data-full"]/table')[0]
    table_2004 = get_national_table(table)
    # mccain vs obama
    url = "http://www.realclearpolitics.com/epolls/2008/president/us/general_election_mccain_vs_obama-225.html"
    xml_tree = get_xml_from_url(url)
    table = xml_tree.xpath('//div[@id="polling-data-full"]/table')[0]
    table_2008 = get_national_table(table)

    url = "http://www.realclearpolitics.com/epolls/2012/president/us/general_election_romney_vs_obama-1171.html"
    xml_tree = get_xml_from_url(url)
    table = xml_tree.xpath('//div[@id="polling-data-full"]/table')[0]
    table_2012 = get_national_table(table)
    return table_2004, table_2008, table_2012

if __name__ == "__main__":
    table_frame = download_latest_state_polls()
    table_frame.to_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2012_poll_data_details.csv", index=False, sep="\t")

    table_2004, table_2008, table_2012 = download_historical_polls()
    table_2004.to_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2004_poll_data.csv", index=False, sep="\t")
    table_2008.to_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2008_poll_data.csv", index=False, sep="\t")
    table_2012.to_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2012_poll_data.csv", index=False, sep="\t")

    state_frame_2012 = download_state_polls()
    table_2012.to_csv("/home/skipper/school/seaboldgit/talks/pydata/data/2012_poll_data_states.csv", index=False, sep="\t")
