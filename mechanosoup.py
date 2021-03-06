# %% Python 3.8
# Author: TristanCB
# Platform: Windows 10
# Code editor: Visual Studio Code
# Environment/package manager miniconda
# Software dependencies: Firefox
# Ensure that geckodriver.exe is included in this project's root dir: https://github.com/mozilla/geckodriver/releases
"""
Mechano-soup.
Machine learning text classification approach based on regular expressions and html structure.
"""
import sys
import numpy as np
import ast
from bs4 import BeautifulSoup
from bs4.element import Comment
import time
import funcy
import datetime
import re
import json
import requests
import pickle
from functools import wraps
import random

from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.options import Options

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, AveragePooling2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.metrics import classification_report, confusion_matrix

from focal_loss import categorical_focal_loss

# %%
# Use headless option if you want to prevent browser from popping up during script's execution.
options = Options()
options.headless = True

# Enter desired URL
url = "https://santemontreal.qc.ca/population/coronavirus-covid-19/depistage-covid-19-a-montreal/"

# We will build up the training set using beautiful soup.
# All will be stored in this dictionary.
html_text_data_frame ={}
# Stores all uniques in url
attributes_all      = []
attrs_values_all    = []

# Ensure that Firefox is installed and that the driver is included: https://github.com/mozilla/geckodriver/releases, https://www.mozilla.org/en-CA/firefox/new/
driver = webdriver.Firefox(options=options, executable_path="./geckodriver.exe")
# Fetch the page... I would like to be able to use a simple request, but intially while testing it did not work.
driver.get(url)
soup = BeautifulSoup(driver.page_source)

# %% Build dataset from html doc

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

def attribute_tolkenizor(tag):
    """
    
    """
    tag_attrs = tag.attrs
    tag_tolkens = []

    for i in tag_attrs:
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
        'sup', 'nav', 'h4', 'button', 'div', 'html', 'b', 'p', 'h2', 'th', 
        'strong', 'sup', 'h3', 'span','input','i','img','style','figure',
        'br','script','iframe','noscript','link','meta','title','base','head',
        'names','svg','g','path'
    ]

    # print(string)
    # Should probably include all of the possible HTML_tags... or a try, except
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
            
            ## vector format
            # tolkens.append([i ,len(ij)])

            ## Linear
            tolkens.append(i)
            tolkens.append(len(ij))

    return tolkens


# Find all the displayed text in the page
texts = soup.findAll(text=True)

visible_texts = filter(tag_visible, texts)
# Construct a list with all of them
import codecs
text_element = [t for t in visible_texts]
for i, ij in enumerate(text_element):
    # pickled = pickle.dumps(ij, 0).decode()
    # pickled = codecs.encode(pickle.dumps(ij), "base64").decode()
    # print(type(pickled))
    # print(len(pickled))
    # assert 1 == 0

    # Remove null text
    if len(ij.strip()) < 2:
        # print("SKKKKKIPED")
        continue
    print(ij)

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
    
    # Get attributes and their values as tolkens
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
    # html_text_data_frame[i]["Data_base64"] = pickled
    html_text_data_frame[i]["length"] = [len(ij)]
    html_text_data_frame[i]["parent_tags"] = [len(parent)]
    html_text_data_frame[i]["name"] = [HTMLtolkenizor(parent.name)]
    html_text_data_frame[i]["nested_structure"] = nested_structure
    html_text_data_frame[i]["attribute_mask"] = attribute_mask
    html_text_data_frame[i]["next"] = next
    html_text_data_frame[i]["previous"] = previous
    html_text_data_frame[i]["re_tolkens"] = regexTokenizor(html_text_data_frame[i]["Data"])

    # # Manually building the dataset
    # print(html_text_data_frame[i]["Data"])
    # class_num = input("Enter class number")
    # if class_num == "exit":
    #     break
    # try:
    #     html_text_data_frame[i]["text_class"] = int(class_num)
    # except ValueError:
    #     print("Not a valid class, assigning 0")
    #     html_text_data_frame[i]["text_class"] = 0
    # # /Manually building the dataset

# 

# %% Store soup to disk
def run_a_train_eval_models(skip_list):

    def get_macro_avg(classification_report):
        return classification_report.split('macro avg')[1].split("      ")[3]

    def benchmark_time(model, X, sample_size = 1):
        """
        Custom function to time predict performance, works for TF models and SKLEARN ones,
        I should consider using timeit to account for garbage collection and etc.
        """
        t = time.process_time()
        for i in range(sample_size):
            model.predict(X)
        average_time_elapsed = (time.process_time() - t)/sample_size
        return average_time_elapsed

    sys.setrecursionlimit(10000)

    # pickle.dump( html_text_data_frame, open( "HTMLDATAFRAME_final_with_raw.p", "wb" ) )



    with open("HTMLDATAFRAME_final.p", 'rb') as pickle_file:
        html_text_data_frame = pickle.load(pickle_file)
    print()

    # %% Prune soup
    # amount_of_null = 0
    # candidate_for_deletion = []
    # ## Check data distribution. and prune off some of the null data...
    # for i in html_text_data_frame:
    #     # We will
    #     if html_text_data_frame[i]["text_class"] == 0:
    #         if random.randrange(10) > 2:
    #             candidate_for_deletion.append(i)
    #             continue
    #         amount_of_null += 1

    # # Do the deletion randomly
    # for i in candidate_for_deletion:
    #     del html_text_data_frame[i]
    # print(amount_of_null)

    ## %% Construct dataset
    import numpy as np
    import tensorflow as tf
    from keras.utils import to_categorical
    Data_X = None
    # The representation of the text element we are trying to extract will be based off of the following
    max_len = 22
    data_vectors = 8

    for i in html_text_data_frame:
        X = []
        for j in html_text_data_frame[i]:
            # Omit placing vertain data in matrix describing text. Useful to find what matters most.
            # Data and text_class must occur in this list.
            # skip_list = ["Data", "text_class" ,"Data_base64", "next", "previous", "attribute_mask", "re_tolkens", "nested_structure", "name", "length"]
            # skip_list = ["Data", "text_class", "Data_base64"]

            if [] != [True for i in skip_list if j == i]:
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
        if X == []:
            X = np.zeros((data_vectors,max_len))
        ## Format for CNN channels last
        # X = np.expand_dims(X, axis=-1)
        
        # Stacks or creates np array
        if Data_X is None:
            Data_X = X
        else:
            Data_X = np.vstack((Data_X,X))
            

    print("Shape of Data_X", Data_X.shape)
    # Data_X_classifier = Data_X.reshape((Data_X.shape[0],Data_X.shape[1]*Data_X.shape[2]))
    # print("Shape of Data_X_classifier", Data_X_classifier.shape)

    # Get text class
    Data_Y = [int(html_text_data_frame[i]["text_class"]) for i in html_text_data_frame]
    Data_Y_categorical = to_categorical(Data_Y)
    print("Shape of Data_Y_categorical", Data_Y_categorical.shape)

    # ## %% Train and validation split 
    # split_value = int(Data_X.shape[0]*0.6)
    # # print(Data_X.shape)
    # X_TRAIN         = Data_X[:split_value]
    # X_VALIDATE      = Data_X[split_value::]

    # Y_TRAIN         = Data_Y_categorical[:split_value]
    # Y_VALIDATE      = Data_Y_categorical[split_value::]

    ## %% Train and validation split random
    X_TRAIN         = []
    X_VALIDATE      = []

    Y_TRAIN         = []
    Y_VALIDATE      = []

    split_VALDATION_ODDS = 0.25
    for i in range(len(Data_X)):
        if random.random() > split_VALDATION_ODDS:
            X_TRAIN.append(Data_X[i])
            Y_TRAIN.append(Data_Y_categorical[i])
        else:
            X_VALIDATE.append(Data_X[i])
            Y_VALIDATE.append(Data_Y_categorical[i])
    
    X_TRAIN         = np.asarray(X_TRAIN)
    X_VALIDATE      = np.asarray(X_VALIDATE)

    Y_TRAIN         = np.asarray(Y_TRAIN)
    Y_VALIDATE      = np.asarray(Y_VALIDATE)

    print(X_TRAIN.shape)
    print(X_VALIDATE.shape)
    print(Y_TRAIN.shape)
    print(Y_VALIDATE.shape)
    ## %% Build the learning machine
    # X_TRAIN = np.expand_dims(X_TRAIN,axis=1)
    # print(X_TRAIN.shape)

    # Idea of padding the vector and fitting it to a DEEP one... 
    ## MobileNet network ##
    # from tensorflow.keras.applications.inception_v3 import InceptionV3
    # from tensorflow.keras.layers import Input
    # MobileNet_shape = (224, 224, 3)
    # input_tensor = Input(shape=MobileNet_shape)

    # def pad_for_(np_array,input_shape):
    #     ''' Reshaping input to be valid for a model '''
    #     padded_array = []
    #     for i, ij in enumerate(np_array):
    #         padded = np.pad(ij, ((0,input_shape[0]-ij.shape[0]) , (0,input_shape[1]-ij.shape[1])), mode = 'constant', constant_values=(0, 0))
    #         three_channel_format = np.asarray([padded,padded,padded])
    #         three_channel_format = np.moveaxis(three_channel_format, 0, -1) 
    #         padded_array.append(three_channel_format)
    #     return np.asarray(padded_array)
    # X_TRAIN_MobileNet = pad_for_(X_TRAIN,MobileNet_shape)
    # X_VALIDATE_MobileNet = pad_for_(X_VALIDATE,MobileNet_shape)
    # # for i, ij in enumerate(X_TRAIN):
    # #     padded = np.pad(ij, ((0,MobileNet_shape[0]-ij.shape[0]) , (0,MobileNet_shape[1]-ij.shape[1])), mode = 'constant', constant_values=(0, 0))
    # #     three_channel_format = np.asarray([padded,padded,padded])
    # #     three_channel_format = np.moveaxis(three_channel_format, 0, -1) 
    # #     print(three_channel_format.shape)
    # #     X_TRAIN_MobileNet.append(three_channel_format)
    # X_TRAIN_MobileNet = np.asarray(X_TRAIN_MobileNet)
    # print(X_TRAIN_MobileNet.shape)
    # print(X_VALIDATE_MobileNet.shape)

    # # this could also be the output a different Keras model or layer    # this could also be the output a different Keras model or layer
    # MobileNet = tf.keras.applications.MobileNet(
    #     input_shape=None,
    #     alpha=1.0,
    #     depth_multiplier=1,
    #     dropout=0.001,
    #     include_top=True,
    #     weights=None,
    #     input_tensor=None,
    #     pooling=None,
    #     classes=Data_Y_categorical.shape[1],
    #     classifier_activation="softmax",
    # )

    # new_model = Sequential()
    # new_model.add(MobileNet)
    # new_model.add(layers.Flatten())
    # new_model.add(layers.Dense(Data_Y_categorical.shape[1], activation='softmax'))
    # print(new_model.summary())
    # new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    # # Callback to prevent overtraining
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # new_model.fit(X_TRAIN_MobileNet, Y_TRAIN, epochs=100, batch_size=64, callbacks=[callback], validation_split=0.1)

    # VALIDATION_fit = new_model.predict(X_VALIDATE_MobileNet)
    # MOBILENET_classification_report = classification_report(VALIDATION_fit.argmax(axis=1), Y_VALIDATE.argmax(axis=1))

    # # VALIDATION MATRIX
    # val_matrix = confusion_matrix(VALIDATION_fit.argmax(axis=1), Y_VALIDATE.argmax(axis=1))
    # print(val_matrix)
    # print(MOBILENET_classification_report)
    ## / MobileNet network ##

    ## Custom LSTM-CNN network
    model = models.Sequential()
    # LSTM_UNITS = 128
    LSTM_UNITS = 64

    model.add(layers.LSTM(LSTM_UNITS, activation='elu', input_shape=(Data_X.shape[1], Data_X.shape[2]), return_sequences=True))
    model.add(tf.keras.layers.Reshape((Data_X.shape[1], LSTM_UNITS, 1)))

    units = 64
    model.add(layers.Conv2D(units, (1,3), activation='elu'))
    model.add(layers.Conv2D(units, (1,3), activation='elu'))
    model.add(layers.Conv2D(units, (1,3), activation='elu'))
    model.add(layers.Conv2D(units, (3,1), activation='elu'))

    model.add(layers.Conv2D(units/2, (1,3), activation='elu'))
    model.add(layers.Conv2D(units/2, (1,3), activation='elu'))
    model.add(layers.Conv2D(units/2, (1,3), activation='elu'))
    model.add(layers.Conv2D(units/2, (3,1), activation='elu'))

    model.add(layers.Conv2D(units/4, (1,3), activation='elu'))
    model.add(layers.Conv2D(units/4, (1,3), activation='elu'))
    model.add(layers.Conv2D(units/4, (1,3), activation='elu'))
    
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='elu'))
    # model.add(layers.Dense(32, activation='elu'))
    # model.add(layers.Dense(16, activation='elu'))

    # Output layer with number of defined text classes
    model.add(layers.Dense(Data_Y_categorical.shape[1], activation='softmax'))

    ## Categorical is probably the way to go
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    # We use focal loss, likely to have highly imbalances dataset, try pruning out null entries, I like to assign them class 
    model.compile(loss=categorical_focal_loss(), optimizer="adam", metrics=['accuracy'])

    print(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # Train the machine
    model.fit(X_TRAIN, Y_TRAIN, epochs=25, batch_size=64, callbacks=[callback])

    ## %% Final evaluation of the model
    scores = model.evaluate(X_VALIDATE, Y_VALIDATE, verbose=0)

    VALIDATION_fit = model.predict(X_VALIDATE)
    
    # Time it
    LSTM_infer_time = benchmark_time(model,X_VALIDATE)

    TRAINING_fit = model.predict(X_TRAIN)
    # print("Accuracy: %.2f%%" % (scores[1]*100))
    LSTM_classification_report = classification_report(VALIDATION_fit.argmax(axis=1), Y_VALIDATE.argmax(axis=1))

    # VALIDATION MATRIX
    val_matrix = confusion_matrix(VALIDATION_fit.argmax(axis=1), Y_VALIDATE.argmax(axis=1))
    print(val_matrix)

    classifier_results = {}
    classifier_results["LSTM_CNN"] = {"macro_avg":get_macro_avg(LSTM_classification_report), "speed":LSTM_infer_time}

    # matrix = confusion_matrix(TRAINING_fit.argmax(axis=1), Y_TRAIN.argmax(axis=1))
    # print(matrix)

    ## Cleanup and pretty print
    # import pprint
    # for i in html_text_data_frame:
    #     for j in ['length','parent_tags','name','nested_structure','attribute_mask','next','previous','re_tolkens']:
    #         try:
    #             del html_text_data_frame[i][j]
    #         except KeyError:
    #             pass
    ## /Cleanup and pretty print
    ## For debugging purposes
    # for i in html_text_data_frame:
    #     if html_text_data_frame[i]['text_class'] == 5:
    #         print(html_text_data_frame[i]['Data'])
    #         # time.sleep(10)
    # pp = pprint.PrettyPrinter(indent=5)
    # pp.pprint(html_text_data_frame)

    # %%
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    Y_TRAIN_SKLEARN = np.expand_dims(np.argmax(Y_TRAIN, axis=1),axis=1)
    X_TRAIN_SKLEARN  = np.reshape(X_TRAIN, (X_TRAIN.shape[0],X_TRAIN.shape[1]*X_TRAIN.shape[2]))
    # print(Y_TRAIN_SVM.shape)
    # print(X_TRAIN_SVM.shape)

    Y_VALIDATE_SKLEARN  = np.expand_dims(np.argmax(Y_VALIDATE, axis=1),axis=1)
    X_VALIDATE_SKLEARN  = np.reshape(X_VALIDATE, (X_VALIDATE.shape[0],X_VALIDATE.shape[1]*X_VALIDATE.shape[2]))
    # print(Y_VALIDATE_SVM.shape)
    # print(X_VALIDATE_SVM.shape)

    classifiers = [
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel = 'linear', C = 1),
        KNeighborsClassifier(n_neighbors = Data_Y_categorical.shape[1]).fit(X_TRAIN_SKLEARN, Y_TRAIN_SKLEARN),
    ]

    ## I should test em all out!
    def eval_classifiers(X = X_TRAIN_SKLEARN, Y = Y_TRAIN_SKLEARN, X_val = X_VALIDATE_SKLEARN, Y_val = Y_VALIDATE_SKLEARN):
        for i in classifiers:
            model = i.fit(X, Y)
            execution_speed = benchmark_time(model, X, sample_size = 1)/len(X)
            # creating a confusion matrix 
            predictions = model.predict(X_val)
            cm = confusion_matrix(Y_val, predictions)
            print(cm)
            # Report
            report = classification_report(Y_val, predictions)
            print(report)
            print(i)
            classifier_results[str(i).strip('()')] = {"macro_avg":get_macro_avg(report), "speed":execution_speed}
    
    eval_classifiers()
    print(classifier_results)
    return classifier_results

skip_list = ["length","parent_tags","name","nested_structure","attribute_mask","next","previous", "re_tolkens"]

import csv
# %%
with open('full_matrix_13.csv', 'w', newline='') as csvfile:

    fieldnames = [
       'Input_removed', 'Model', 'macro_avg', 'speed'
        ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # for i in skip_list:
    for i in range(64):
        for k in range(1):
            ignore_skip_list = ["Data", "text_class", "Data_base64"]
            ## Only one data
            # ignore_skip_list += skip_list
            # ignore_skip_list.pop(ignore_skip_list.index(i))
            ## All but removing one
            # ignore_skip_list.append(i)
            
            print(ignore_skip_list)
            results = run_a_train_eval_models(ignore_skip_list)
            print(results)
            
            for j in results:
                writer.writerow({
                    'Input_removed': i,
                    'Model': j,
                    'macro_avg': results[j]["macro_avg"],
                    'speed': results[j]["speed"],
                    })

# import os
# os.system('shutdown -s')

# %% Next steps,
# Process pages with model.predict(...) 
# could be used to find specific sequences.

# Copyright TristanCB
