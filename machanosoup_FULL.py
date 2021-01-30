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
url=""

# Ensure that Firefox is installed and that the driver is included: https://github.com/mozilla/geckodriver/releases, https://www.mozilla.org/en-CA/firefox/new/
driver = webdriver.Firefox(options=options, executable_path="./geckodriver.exe")
# Fetch the page... I would like to be able to use a simple request, but intially while testing it did not work.
driver.get(url)
soup = BeautifulSoup(driver.page_source)

# %%
# A statistical approach to HTML document text classification
from bs4.element import Comment

html_text_data_frame ={}


def tag_visible(element):
    """
    https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text, (Jbochi, 2009)
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def class_and_id(tag):
    """
    inspired from bs4docs
    """
    # print(type(tag))
    if tag.has_attr('class') and tag.has_attr('id'):
        return 0
    elif tag.has_attr('class') and not tag.has_attr('id'):
        return 1
    elif tag.has_attr('id') and not tag.has_attr('class'):
        return 2
    else:
        return 3

# Stores all uniques in url
attributes_all      = []
attrs_values_all    = []
def attribute_tolkenizor(tag):
    """
    
    """
    tag_attrs = tag.attrs
    tag_tolkens = []

    for i in tag_attrs:
        # print("$$$")
        # print(i)
        # print(tag_attrs[i])
        # print("$$$")
        try:
            attr_index = attributes_all.index(i)
        except ValueError:
            attributes_all.append(i)
            attr_index = attributes_all.index(i)

        for j in tag_attrs[i]:
            try:
                value_index = attrs_values_all.index(j)
                tag_tolkens.append([attr_index,value_index])
            except ValueError:
                attrs_values_all.append(j)
                value_index = attrs_values_all.index(j)
                tag_tolkens.append([attr_index,value_index])
    if tag_tolkens == []: return
    return tag_tolkens
    
    # print(zero_padding_3d(x, p=1)[:, :, 0])
    # print(np.expand_dims(np.asarray(tag_tolkens), axis=0).shape)


def HTMLtolkenizor(string):
    HTML_tags = [
        
        '[document]', 'label', 'footer', 'ul', 'td', 'tbody', 'li', 
        'article', 'main', 'h2', 'a', 'th', 'header', 'form', 'thead', 
        'section', 'body', 'span', 'table', 'h1', 'tr', 'h3', 'strong', 
        'sup', 'nav', 'h4', 'button', 'div', 'html', 'b', 'p', 'h2', 'th', 'strong', 'sup', 'h3',
        'span','input','i','img','style','figure','br','script','iframe','noscript','link','meta','title','base','head',
        'names','svg','g','path'
    ]

    # print(string)
    return HTML_tags.index(string)

def regexTokenizor(string):
    reg_exs = 0
    """
    Should probably use NLP tolkenizer.
    returns a list of integer tolkens
    """
    phone                       = re.findall(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", string)
    postal_code                 = re.findall(r'\b(?!.{0,7}[DFIOQU])[A-VXY]\d[A-Z][^-\w\d]\d[A-Z]\d\b', string)
    word_digit_white_space      = re.findall(r'\w \d \s', string)
    digit                       = re.findall(r'\d', string)
    digits                      = re.findall(r'\d^', string)
    chars                       = re.findall(r'\w', string)
    not_word_digit_white_space  = re.findall(r'\w \d \s', string)
    common_email_ids            = re.findall(r'/^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*$/', string)
    
    regex_list = [
        postal_code,
        word_digit_white_space,
        digit,
        digits,
        chars,
        not_word_digit_white_space,
        common_email_ids,
    ]

    tolkens = []

    for i, ij in enumerate(regex_list):
        if len(ij) > 0:
            # print(ij)
            # print("REGEXMATCH FOUND")
            reg_exs += 1 # Counter
            # time.sleep(5)
            
            ## vector format
            # tolkens.append([i ,len(ij)])

            ## Linear
            tolkens.append(i)
            tolkens.append(len(ij))

    return tolkens

def sip_from_soup(soup):
    # Find all the displayed text in the page
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    # Construct a list with all of them
    text_element = [t for t in visible_texts]
    for i, ij in enumerate(text_element):
        # Remove null text
        if len(ij.strip()) < 2:
            # print("SKKKKKIPED")
            continue

        # print(type(ij))
        # print(i)

        # Nested dataframe
        html_text_data_frame[i] = {}

        # Only use the a set amount of sideways tags
        sideways_limit = 5
        next = [HTMLtolkenizor(i.name) for i in ij.parent.find_all_next()][0:sideways_limit]
        previous = [HTMLtolkenizor(i.name) for i in ij.parent.find_all_previous()][0:sideways_limit]
        #
        # Get its nesting.
        nested_structure = [HTMLtolkenizor(i.name) for i in ij.find_parents()]
        attribute_mask = [class_and_id(i) for i in ij.find_parents()]
        attributes = [attribute_tolkenizor(i) for i in ij.find_parents()]
        
        # 
        attrs_classes_tolkens = []
        attrs_value_tolkens = []
        # print(attributes)
        for j in attributes:
            if j is None:
                continue
            for k in j:
                attrs_classes_tolkens.append(k[0])
                attrs_value_tolkens.append(k[1])

        assert len(attrs_classes_tolkens) == len(attrs_value_tolkens)

        parent = ij.findParent()
        html_text_data_frame[i]["Data"] = ij
        html_text_data_frame[i]["length"] = [len(ij)]
        html_text_data_frame[i]["parent_tags"] = [len(parent)]
        html_text_data_frame[i]["name"] = [HTMLtolkenizor(parent.name)]
        html_text_data_frame[i]["nested_structure"] = nested_structure
        html_text_data_frame[i]["attribute_mask"] = attribute_mask
        html_text_data_frame[i]["next"] = next
        html_text_data_frame[i]["previous"] = previous
        html_text_data_frame[i]["re_tolkens"] = regexTokenizor(html_text_data_frame[i]["Data"])

        print(html_text_data_frame[i]["Data"])
        # print(np.expand_dims(np.asarray(html_text_data_frame[i]["re_tolkens"]), axis = 0).shape)

        # Manually building the dataset
        class_num = input("Enter class number")
        if class_num == "exit":
            break
        try:
            html_text_data_frame[i]["text_class"] = int(class_num)
        except ValueError:
            print("Not a valid class, assigning 0")
            html_text_data_frame[i]["text_class"] = 0
        # /Manually building the dataset

# 
sip_from_soup(soup)

# %% Store soup to disk
import pickle
import sys
sys.setrecursionlimit(10000)
# pickle.dump( html_text_data_frame, open( "HTMLDATAFRAME.p", "wb" ) )
with open("HTMLDATAFRAME_test.p", 'rb') as pickle_file:
    html_text_data_frame = pickle.load(pickle_file)
print()
# %% Prune soup
import random
amount_of_null = 0
candidate_for_deletion = []
## Check data distribution. and prune off some of the null data...
for i in html_text_data_frame:
    # We will
    if html_text_data_frame[i]["text_class"] == 0:
        if random.randrange(10) > 2:
            candidate_for_deletion.append(i)
            continue
        amount_of_null += 1

for i in candidate_for_deletion:
    del html_text_data_frame[i]
print(amount_of_null)

# %% Construct dataset
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
Data_X = None
max_len = 22
data_vectors = 8
for i in html_text_data_frame:
    X = []
    for j in html_text_data_frame[i]:
        if "Data" == j or "text_class" == j:
            continue

        if html_text_data_frame[i][j] is None:
            # print("skipped")
            x.append([0 for i in range(max_len)])
            continue
        while len(html_text_data_frame[i][j]) < max_len:
            html_text_data_frame[i][j].append(0)
        X.append(html_text_data_frame[i][j])

    X = np.expand_dims(np.asarray(X),axis=0)

    # Fixes an issue where all are None 
    print(X)
    print(len(X))
    # time.sleep(3)
    if X == []:
        print("$$$$$$$$$$$")
        X = np.zeros((data_vectors,max_len))
    ## Format for CNN channels last
    # X = np.expand_dims(X, axis=-1)
    
    # Stacks or creates np array
    if Data_X is None:
        Data_X = X
    else:
        Data_X = np.vstack((Data_X,X))
        

print(Data_X.shape)
# Data_X_Classifier = np.concatenate(Data_X, axis=0)
print(Data_X.reshape((247,176)).shape)
# print(Data_X_Classifier.shape)
Data_X_classifier = Data_X.reshape((247,176))

Data_Y = [html_text_data_frame[i]["text_class"] for i in html_text_data_frame]
Data_Y_categorical = to_categorical(Data_Y)

from focal_loss import categorical_focal_loss
# print(data_X.shape)

# %%
## ML classification approach:
from sklearn.metrics import classification_report, confusion_matrix

model = models.Sequential()
model.add(layers.LSTM(96, activation='relu', input_shape=(8, 22)))
model.add(layers.Dense(96, activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(Data_Y_categorical.shape[1], activation='softmax'))

model.compile(loss=categorical_focal_loss(), optimizer="adam", metrics=['accuracy'])

print(model.summary())
model.fit(Data_X, Data_Y_categorical, epochs=5, batch_size=64, validation_split= 0.1)
# Final evaluation of the model
scores = model.evaluate(Data_X, Data_Y_categorical, verbose=0)
X_train_fit = model.predict(Data_X)
print("Accuracy: %.2f%%" % (scores[1]*100))

matrix = confusion_matrix(X_train_fit.argmax(axis=1), Data_Y_categorical.argmax(axis=1))
print(matrix)

# %%
### Post processing functions ###
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
names.append("Centre universitaire")
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

assert clinics[1]["name"] == "Clinique de dépistage de la COVID-19"
assert clinics[1]["rdzvs"] == "Sans rendez-vous"
assert clinics[1]["opening_hours"] == "Tous les jours* De 8 h à 19 h 45 *Le 25 décembre et le 1er janvier 2021: De 8 h à 16 h"

# Save the answer for question part 2
with open('covid_clinics_wt.json', 'w',  encoding="utf-8") as fp:
    json.dump(clinics, fp,  indent=4, ensure_ascii=False)

## Bonus # 2
# %% BONUS 2...
## For this question it is said. to use the following input.
# Input:
# Sam. et dim. 10 h à 17 h
# I decided to take this one step further and try and process the actual string describing opening hours in french.
# Warning the following code did work but is extremely messy. 
 
# TESTSTRING = 'Tous les jours* De 8 h à 20 h *Les 24, 25, 26, 27 et 31 décembre et le 1er janvier 2021: De 8 h à 16 h'
# TESTSTRINGS = [clinics[i]["opening_hours"] for i in clinics]

def times_from_french_string(opening_times_string):
    """
    Processes a string of the format: 'Tous les jours* De 8 h à 20 h *Les 24, 25, 26, 27 et 31 décembre et le 1er janvier 2021: De 8 h à 16 h'
    to it's respective openings for upcoming days
    """
    french_months_full = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    english_months_GMT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Assign calendar dates to days of the week
    days_full = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    days_GMT = ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun" ]

    # I justify using split with an index referencing the date. GMT timestamps are standardized.
    # split_time = wait_times.headers["Date"].split()
    today = datetime.datetime.today()
    # According to GMT header. 
    # day_index = days_GMT.index(split_time[0].replace(",","")) # Fund the corresponding say
    day_index = today.weekday()
    ## 
    upcoming_days = []

    while len(upcoming_days) != 6:
        if len(upcoming_days) == 0:
            day_date = str(datetime.datetime.today())
        else:
            day_date = str(datetime.datetime.today() + datetime.timedelta(days=len(upcoming_days)))
        
        day_month = int(day_date.split("-")[1])
        day_date =  day_date.split(" ")[0].split("-")[-1]
        
        # print(day_date)
        # print(day_month)

        upcoming_days.append((days_full[day_index], day_date, french_months_full[day_month-1]))
        day_index += 1
        if day_index == 7:
            day_index = 0

    ##
    # Gets the corresponding index of the month from datetime today
    current_month_index = int(str(today).split("-")[1])-1
    # print(current_month_index)
    current_month = french_months_full[current_month_index]
    # Handles Dec to Jan transition
    if current_month_index == 11:
        next_month = french_months_full[0]
    else:
        next_month = french_months_full[current_month_index + 1]

    def format_time_string(string):
        digits = [int(s) for s in string.split() if s.isdigit()]
        # print(digits)
        if len(digits) == 1:
            return f"{digits[0]}:00"
        elif len(digits) >= 2:
            return f"{digits[0]}:{digits[1]}"
        else:
            # assert 1 == 2
            pass

    def find_date_exceptions(string, current_month = current_month, next_month = next_month):
        """
        Input a string seperated by to handle the dates with differing openings
        Returns dates for which times are not the same.
        """
        date_exceptions = []

        attribute_hours = True

        if "Samedi et dimanche" in string:
            return
        
        if "Fermé" in string:
            attribute_hours = False

        if "Le" in string:
            if attribute_hours == True:
                # print("PTTTTTERNFOUND: colon seperated")
                openings_split = string.split(":")[-1].split("à")
                open    = format_time_string(openings_split[0])
                close   = format_time_string(string.split(":")[-1].split("à")[1])
                # print(open,close)
                formatted_open_close_times = [{"open": open, "close": close}]
            else:
                formatted_open_close_times = ["CLOSED"]

            current_month_data = string.split(current_month)[0].replace(",","").replace("er","")
            next_month_data = string.split(current_month)[1].split(next_month)[0].replace(",","").replace("er","")
            ## Construct a dict with exception dates
            # print([int(s) for s in i.split(current_month)[0].strip(",").split() if s.isdigit()])
            date_exceptions = [(current_month, int(s), formatted_open_close_times) for s in [int(s) for s in current_month_data.split() if s.isdigit()]]
            if next_month in string:
                [date_exceptions.append((next_month, int(s), formatted_open_close_times)) for s in [int(s) for s in next_month_data.split() if s.isdigit()]]

        return date_exceptions

    # Im not going to try and deal with these just yet. 
    # If À compter is in the string i will just ignore for now
    if "À compter" in opening_times_string:
        print("UNHANDLED EXCEPTION NOT ADDING TIME SLOTS FOR THESE CLINICS")
        return
    
    # I want to return the following dict.
    openings = {
        "monday":"",
        "tuesday":"",
        "wednesday":"",
        "thursday":"",
        "friday":"",
        "saturday":"",
        "sunday":"",
    }

    # The basis of the processing strats here
    split_by_asterics = opening_times_string.split("*")
    # print(split_by_asterics)

    if len(split_by_asterics) > 1:
        string_to_process =  split_by_asterics[1]
    elif len(split_by_asterics) == 1:
        string_to_process =  split_by_asterics[0]

    openings_split = string_to_process.split(":")[-1].split("à")
    open    = format_time_string(openings_split[0])
    close   = format_time_string(string_to_process.split(":")[-1].split("à")[1])
    # print(open,close)
    formatted_open_close_times = [{"open": open, "close": close}]
    # print(formatted_open_close_times)

    # Process main opening times
    if "Tous" in split_by_asterics[0]:
        # print("ALLLL day eeer day")
        for i in openings:
            openings[i] = formatted_open_close_times

    elif "Du lundi au vendredi" in split_by_asterics[0]:
        # print("Weekends are for the fam")
        for i in openings:
            if i == "saturday" or i == "sunday":
                openings[i] = "CLOSED"
                continue
            else:
                openings[i] = formatted_open_close_times

    elif "Samedi et dimanche" in split_by_asterics[0]:
        # print("Weekend warrior")
        for i in openings:
            if "saturday" not in i or "sunday" not in i:
                openings[i] = "CLOSED"
                continue
            else:
                openings[i] = formatted_open_close_times

    # DEAL WITH EXCEPTIONs
    exceptions = []
    # Process each list item after splitting by asterics and append them to a list
    for i in split_by_asterics:
        exception_block = (find_date_exceptions(i))
        if exception_block is not None:
            exceptions += exception_block
    # print(exceptions)

    # For each upcoming day check if an exception exists. If so override openings.
    for i in upcoming_days:
        # print(i)
        # For each entry in the upcomings days which has a matchinf date and month,
        # update it to what it should be.
        for j in exceptions:
            if j[0] == i[2]:
                if int(j[1]) == int(i[1]):
                    openings[i[0]] = j[2]
                    # print("MMMMMAAAATCH FOUND")

    return openings

# PRocess all clinics
for clinic_id in clinics:
    easy_openings = times_from_french_string(clinics[clinic_id]["opening_hours"])
    clinics[clinic_id]["easy_openings"] = easy_openings


print(clinics)

# Unit test for bonus question
# Should probably increase the coverage for this test but gives an idea.
# assert clinics[1]["easy_openings"]["tuesday"] == [{'open': '8:00', 'close': '19:45'}]

# Save the answer for Bonus question
## TypeError: 'str' object is not callable??
with open('covid_clinics_wt_easy.json', 'w',  encoding="utf-8") as fp:
    json.dump(clinics, fp,  indent=4, ensure_ascii=False)

# %% regular expression AWS lambda API
# This is an API I created using AWS lambda for the re expression machings I found doing this coding exercise.

api = " https://7m8n6jg4wf.execute-api.us-east-2.amazonaws.com/default/string_processing"
postal_code_example = {
    "pattern":"postal_code",
    "string_input":"hello AWS K8X 2N9 dolphin H7Y 8K0",
    }

phone_number_example = {
    "pattern":"phone_number",
    "string_input":"Hey H9X 2N7 518-966-8988 anaconda H7Y 8K0",
    }

api_tests = [postal_code_example,phone_number_example]
for i in api_tests:
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(api, data=i, headers=headers)
    print(r.text)

##################### Closing remarks ###########################
"""
Another interesting challenge I tried was to get headless selenium to work using AWS lambda, It is doable.
Most people report having to use chrome binary. I was not able to get it function tho, I would have to
reconsider my approach and likely learn how to properly use docker and AWS CLI interface...
Alternatively PhantomJS might be easier to implement in AWS lambda.

I also tried to apply machine learning for text extractions but the extramely small training data I was
able to gather (i.e. from copy/pasting manually) was not sufficient. 
I believe this still could be done using an extreme amount of data augmentation.
Nevertheless this might not result in anything practical just thought I'd give it a try.

A more interesting approach would be to use Natural language processing or regular expression matching to extract desired info from an html,
This could then be used to build up a training dataset having the html tag structures as input with a corresponding output
of 0 and 1 (text labeled interesting and text labeled not interesting). I think that using this approach might
be more efficient than pure expression matching for data mining. 

Thanks for the opportunity!

Have a good day.
"""
# Copyright TristanCB
