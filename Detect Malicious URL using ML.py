import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

# Data Pre processing
print(os.listdir(r'C:\Users\Eliran Genasia\Documents\work\Preception point'))
urldata = pd.read_csv(r'C:\Users\Eliran Genasia\Documents\work\Preception point\urldata.csv\urldata.csv')
print(urldata.head())

# Removing the unnamed columns as it is not necesary.
urldata = urldata.drop('Unnamed: 0',axis=1)
print(urldata.head())
print(urldata.shape)
print(urldata.info())

# Checking for missing values
print(urldata.isnull().sum())

"""
The following features will be extracted from the URL for classification.

Length Features:
    Length Of Url
    Length of Hostname
    Length Of Path
    Length Of First Directory
    Length Of Top Level Domain

Count Features:
    Count Of '-'
    Count Of '@'
    Count Of '?'
    Count Of '%'
    Count Of '.'
    Count Of '='
    Count Of 'http'
    Count Of 'www'
    Count Of Digits
    Count Of Letters
    Count Of Number Of Directories

Binary Features:
    Use of IP or not
    Use of Shortening URL or not
Apart from the lexical features, we will use TFID - Term Frequency Inverse Document as well.
"""
# This needs to be run in the terminal: pip install tld

# Importing dependencies
from urllib.parse import urlparse
from tld import get_tld

# Length Features

# Length of URL
urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))

# Hostname Length
urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))

# Path Length
urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))

# First Directory Length


def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0


urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))

# Length of Top Level Domain
urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i, fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1


urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))

print(urldata.head())
urldata = urldata.drop("tld", 1)
print(urldata.head())

# Count Features

urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))
urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))
urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')


urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))

print(urldata.head())

# Binary Features
import re


def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return -1
    else:
        return 1


urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))


# Return -1 if there was a use of shortening url else return 1
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1


urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))
print(urldata.head())

# Data Visualization

# Heatmap
corrmat = urldata.corr()
f, ax = plt.subplots(figsize=(25,19))
sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':10})
plt.show()

# Number of benign and malicious url in the data set
plt.figure(figsize=(15,5))
sns.countplot(x='label', data=urldata)
plt.title("Count Of URLs", fontsize=20)
plt.xlabel("Type Of URLs", fontsize=18)
plt.ylabel("Number Of URLs", fontsize=18)
plt.show()

print("Percent Of Malicious URLs:{:.2f} %".format(len(urldata[urldata['label']=='malicious'])/len(urldata['label'])*100))
print("Percent Of Benign URLs:{:.2f} %".format(len(urldata[urldata['label']=='benign'])/len(urldata['label'])*100))

# URL number by url length
plt.figure(figsize=(20,5))
plt.hist(urldata['url_length'], bins=50,color='LightBlue')
plt.title("URL-Length", fontsize=20)
plt.xlabel("Url-Length", fontsize=18)
plt.ylabel("Number Of Urls", fontsize=18)
plt.ylim(0, 10000)
plt.show()

# URL number by hostname length
plt.figure(figsize=(20,5))
plt.hist(urldata['hostname_length'],bins=50,color='Lightgreen')
plt.title("Hostname-Length",fontsize=20)
plt.xlabel("Length Of Hostname",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)
plt.show()

# URL number by count of directories
plt.figure(figsize=(15,5))
plt.title("Number Of Directories In Url",fontsize=20)
sns.countplot(x='count_dir',data=urldata)
plt.xlabel("Number Of Directories",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)
plt.show()

# URL number by count of directories split to benign and malicious
plt.figure(figsize=(15,5))
plt.title("Number Of Directories In Url",fontsize=20)
sns.countplot(x='count_dir',data=urldata,hue='label')
plt.xlabel("Number Of Directories",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)
plt.show()

# URL number by use of ip
plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)

sns.countplot(urldata['use_of_ip'])
plt.ylabel("Number of URLs",fontsize=18)
plt.show()

# URL number by use of ip split to benign and malicious
plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylabel("Number of URLs",fontsize=18)
sns.countplot(urldata['use_of_ip'],hue='label',data=urldata)
plt.ylabel("Number of URLs",fontsize=18)
plt.show()

# Use of http in url
plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylim((0,1000))
sns.countplot(urldata['count-http'])
plt.ylabel("Number of URLs",fontsize=18)
plt.show()

# Use of www in url split to benign and malicious
plt.figure(figsize=(15,5))
plt.title("Use Of WWW In URL",fontsize=20)
plt.xlabel("Count Of WWW",fontsize=18)

sns.countplot(urldata['count-www'],hue='label',data=urldata)
plt.ylim(0,1000)
plt.ylabel("Number Of URLs",fontsize=18)
plt.show()

# Building Models Using Lexical Features Only
"""
I will be using three models for my classification.
1. Logistic Regression
2. Decision Trees
3. Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

# Predictor Variables
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http', 'count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

# Target Variable
y = urldata['result']

print(x.shape)
print(y.shape)

# Splitting the data into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

dt_predictions = dt_model.predict(x_test)
print("accuracy_score of Decision Tree", accuracy_score(y_test, dt_predictions))
print(confusion_matrix(y_test,dt_predictions))

# Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_predictions = rfc.predict(x_test)
print("accuracy_score of Random Forest", accuracy_score(y_test, rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)

log_predictions = log_model.predict(x_test)
print("accuracy_score of Logistic Regression", accuracy_score(y_test, log_predictions))
print(confusion_matrix(y_test, log_predictions))

