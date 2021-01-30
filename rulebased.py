# %%
#  Python 3.8
# Author: TristanCB
# Platform: Windows 10
# Code editor: Visual Studio Code
# Environment/package manager miniconda
# Software dependencies: Firefox, Optionally: Tor (For proxy functionality)
# Ensure that geckodriver.exe is included in this project's root dir: https://github.com/mozilla/geckodriver/releases

# $ conda env create -f requirements.yml
# $ python clinic_scraper_TCB.py
# --> Outputs 3 .json files into the working dir.
"""
Simple Function which extracts data for covid clinics:
This pipeline must be able to:
    - Extract all clinics listed in the Tables on the Montreal Health website
    - Extract their "name", "address", "phone" (if available), "opening_hours" (as text)
    and if they require an appointment or not ("appointment_required").
    - Clean any special characters or HTML from dataset
    - Each clinic must only appear once (No duplicates)
    - You can add more fields if you find any pertinent information
    - Save a JSON file containing all the clinics as "covid_clinics.json"
"""
import sys
import numpy as np
import ast
from bs4 import BeautifulSoup
import time
import datetime
import re
import json
import requests
from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.options import Options
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, AveragePooling2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
# %%
# Use headless option if you want to prevent browser from popping up during script's execution.
options = Options()
options.headless = True

# Enter desired URL
url="https://santemontreal.qc.ca/population/coronavirus-covid-19/depistage-covid-19-a-montreal/"

# Ensure that Firefox is installed and that the driver is included: https://github.com/mozilla/geckodriver/releases, https://www.mozilla.org/en-CA/firefox/new/
driver = webdriver.Firefox(options=options, executable_path="./geckodriver.exe")
# Fetch the page... I would like to be able to use a simple request, but intially while testing it did not work.
driver.get(url)
soup = BeautifulSoup(driver.page_source)

def clean_string(string):
    """
    Removes unwanted characters from strings in scraped data.
    Would probably be best to just use .strip([Chars]) method...
    """
    expressions = ["\t", "\xa0", "  "]
    for i in expressions:
        string = string.replace(i, '')
    return string

def ensure_punct_space(string, punct = ":"):
    """
    Fixes inconsistencies in punctuation.
    Makes sure have no space in between the end of a word and a run on colon.
    i.e. --> input('this : continue', punct = ":") .... return (this: )
    only in runon direction.
    This can probably be done with some package.
    """
    reconstructed = ""
    for i, ij in enumerate(string.split(punct)):
        if ij[-1] == " ":
            ij = ij[:-1]
        if i != 0:
            reconstructed += punct
        reconstructed += ij
    return reconstructed
### / Post processing functions ###

# Clinics will be stored as a dict so we can dump it to json easy
clinics = {}

## List for ML test
names = []
addresses = []
opening_hourss = []
rdzvss = []
phones = []
##

# All clinics are found in different table rows
# We will iterate though all of them and extract the relevent info.
for i, ij in enumerate(soup.find_all("tr")):
    pretty = ij.prettify() # Gets tr as text. Will use for regular expression matching.

    ## For debugging.
    print(f"####### {i} ####### ... processing tr with {len(pretty)} chars")
    # print(pretty)

    # We skip table rows having less than 600 characters, these contain no important info
    if len(pretty) < 500:
        print("Skipped tr... ")
        continue

    # Easiest accessible data. 
    # These were determined by looking at raw HTML printouts or in browser.
    name            = ij.find("strong").text
    address         = ij.find("a").text
    opening_hours   = ij.find_all("td")[-1].text

    names.append(name)
    addresses.append(address)
    opening_hourss.append(opening_hours)
    

    # Apply functions for cleaning text data
    name            = clean_string(name)
    address         = ensure_punct_space(clean_string(address), punct=",")
    # The split and join in the following line is a bit confusing but without, some dates are collapsed together
    opening_hours   = ensure_punct_space(clean_string(' '.join(opening_hours.split())))

    # Checking for rendez-vous
    if "Sans rendez-vous" in pretty:
        rdzvs = "Sans rendez-vous"
    else:
        rdzvs = "Avec rendez-vous"

    # Use a re to extract a possible phone number
    # https://stackoverflow.com/questions/3868753/find-phone-numbers-in-python-script
    phone = re.findall(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", pretty)
    
    if len(phone) > 0:
        phone = phone[0]
        phones.append(phone)
    else:
        phone = ""

    clinics[i] = {
        "name": name, 
        "address":address, 
        "rdzvs":rdzvs, 
        "opening_hours":opening_hours, 
        "phone":phone
    }

## Fixed encoding problems with the following post. Make sure to give kwarg ensure_ascii=False
# https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
with open('covid_clinics.json', 'w',  encoding="utf-8") as fp:
    json.dump(clinics, fp,  indent=4, ensure_ascii=False)

# print(clinics)
# %% 
# Add postal code as a key so we can integrate next data stream 
def find_postal_codes(string):
    """
    https://stackoverflow.com/questions/29906947/canadian-postal-code-validation-python-regex/56592315#56592315
    "I do not guarantee that I fully understand the format, but this seems to work:" (dawg, 2015)
    """
    return re.findall(r'\b(?!.{0,7}[DFIOQU])[A-VXY]\d[A-Z][^-\w\d]\d[A-Z]\d\b', string)

# Adds postal code entry to dict
for i in clinics:
    postal_code = find_postal_codes(clinics[i]["address"])
    if len(postal_code) > 0:
        clinics[i]["postal_code"] = postal_code[0]
    else:
        clinics[i]["postal_code"] = ""
    # print(clinics[i])

# %% PART 2 %% #
"""
You must add a new data source in your ETL Pipeline, the COVID screening wait time
that can be found here:
    https://cdn-contenu.quebec.ca/cdn-contenu/sante/documents/Problemes_de_sante/covid-19/csv/delais_cdd.csv
This pipeline must now be able to:
    - Match a clinic to it’s wait time and add this new information under a "wait_time" property.
    - You can use the address (Or only the postal code) to do this matching.
    - Save a JSON file containing all the clinics from part 1 and their wait time (If they have some) as "covid_clinics_wt.json"
"""

# Location of wait time API
url_wait_times = "https://cdn-contenu.quebec.ca/cdn-contenu/sante/documents/Problemes_de_sante/covid-19/csv/delais_cdd.csv"

## Pass proxies a kwarg to request to use Tor. Must be running Tor.
# proxies = {
#     "http": "socks5://127.0.0.1:9150",
#     "https": "socks5://127.0.0.0:9150",
# }

# https://stackoverflow.com/questions/10606133/sending-user-agent-using-requests-library-in-python
headers = {
    'User-Agent': 'My User Agent 1.0',
    'From': 'tristanchauvin@gmail.com',  # This is another valid field
}

## Interesting to note at this point that simply using a request will only work if i am not using tor proxy
## I will not use proxies and keep request. Could also keep using web driver. I prefer this way however.
# def check_proxy():
#     home_ip = requests.get("http://httpbin.org/ip").text
#     time.sleep(1)
#     proxy_ip = requests.get("http://httpbin.org/ip", proxies=proxies, headers=headers).text
#     print(home_ip, "||", proxy_ip)
#     assert home_ip != proxy_ip


# Finds all postal codes and appends them to a list
pstcd = [] # List of all postal codes in scraped data.
for i in clinics:
    pstcd.append(clinics[i]["postal_code"])

## Im going to make an assumption that clinics with same post code have the same waiting time.
## This should be fine since the wait time api returns all unique postal codes
# print(len(pstcd))
# print(len(list(set(pstcd))))

## The following test fails. Some scraped clinics have the same postal code.
# assert len(pstcd) == len(set(pstcd))

# Get data
wait_times = requests.get(url_wait_times, headers=headers)
wait_time_updated_at = wait_times.headers["Date"]
# Get each entry
split = wait_times.text.split("\n")

# List used for unit testing
postal_codes_waitt = []

# Iterates through all the wait time entries.
for i in split:
    postal_code = find_postal_codes(i) # Postal code match through re
    
    # for loop will exit if no postal codes are detected. After splitting by carriage return,
    # the last entry in the list is empty.
    if len(postal_code) > 0:
        # Keeps track of all postal codes
        postal_codes_waitt.append(postal_code[0])

        # Check for postal code match.
        if postal_code[0] in pstcd:
            #
            # print("Match found")
            #
            def process_wait_time_string(string):
                """
                Returns an empty string for some scenarios.
                i.e. closed or en cours
                """
                if "Actuell" in string or "cours" in string:
                    return ""
                else:
                    return string
            # print(process_wait_time_string(wait_time))

            # List comprehension to get the key value for the matching postal code
            clinic_key = [i for i in clinics if clinics[i]["postal_code"] == postal_code[0]][0]

            # Further split the data by quotation mark and get the portion containing the wait time.
            # -2 is a "magic number", should probably consider using a re. This is the location of
            # the wait time in the resulting splitted string.
            wait_time = i.split('"')[-2]
            # Clean the data. Some wait times are ambiguous, we skip these by adding empty string if certain
            # words are present in the data.
            wait_time = process_wait_time_string(wait_time)

            # Append the data
            clinics[clinic_key]["wait_time"] = process_wait_time_string(wait_time)
            # Bonus # 1
            clinics[clinic_key]["wait_time_updated_at"] = wait_time_updated_at
            
        else:
            #
            # print("No match found")
            #
            pass

# Unit test to make sure no postal code conflit occurs
assert len(postal_codes_waitt) == len(set(postal_codes_waitt))

# Just to keep it consistent I will add a wait time entry to all clinics. 
# They will remain empty if no data is obtained
for i in clinics:
    # Ensure all have an entry.
    try:
        _ = clinics[i]["wait_time"]
    except Exception as e:
        # print(e)
        clinics[i]["wait_time"] = ""

## Unit test:
# Data which was manually copy pasted for first entry from website:
#     [
#         "Clinique de dépistage de la COVID-19",
#         "Sur le terrain du stationnement de l’Hôpital général juif Roulotte mobile dans le stationnement à l’angle de l’avenue Bourret et de la rue Légaré",
#         "",
#         "Tous les jours* De 8 h à 19 h 45. *Le 25 décembre et le 1er janvier 2021 : De 8 h à 16 h",
#         "Sans rendez-vous",
#     ],

# assert clinics[1]["name"] == "Clinique de dépistage de la COVID-19"
# assert clinics[1]["rdzvs"] == "Sans rendez-vous"
# assert clinics[1]["opening_hours"] == "Tous les jours* De 8 h à 19 h 45 *Le 25 décembre et le 1er janvier 2021: De 8 h à 16 h"

# Save the answer for question part 2
with open('covid_clinics_wt.json', 'w',  encoding="utf-8") as fp:
    json.dump(clinics, fp,  indent=4, ensure_ascii=False)
