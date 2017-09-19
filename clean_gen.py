import pandas as pd
import json
import requests
import ast
import re
import nltk
import numpy as np
#from fuzzywuzzy import fuzz
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import lang_detect_exception
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0 # prevent langdetect from producing spurious results between runs
pd.options.display.max_rows = 9999 # list more rows for manual parsing in Pandas
pd.options.display.max_colwidth = 500 # list row content wider

tweets = pd.read_json('./rawData/tweets_gen.json',orient='records',lines=True)

# drop id column
tweets.drop('_id', axis=1, inplace=True)

# set column type to string
tweets['body'] = tweets['body'].astype(str)
tweets['cites_papers'] = tweets['cites_papers'].astype(str)

# make columns format list-like so we can parse with ast
tweets['cites_papers'] = tweets['cites_papers'].str.replace('{\'\$numberLong\':','').str.replace('\'','').str.replace('}','').str.replace(' ','')

# cleanup duplicates
tweets.drop_duplicates(['body'], keep='last', inplace=True)
tweets.reset_index(drop=True, inplace=True)

# remove URLS and shitty typed URLs missing semicolons as well as http(s) stragglers
tweets['body'].replace(re.compile(r"http.?:?//[^\s]+[\s]?"), "", inplace=True)
tweets['body'].replace(re.compile(r"http.?:?/?/?"), "", inplace=True)

# remove handles

tweets['body'] = tweets['body'].apply(nltk.tokenize.casual.remove_handles)

# reduce characters that repeat more than 3 times to 3 instances

tweets['body'] = tweets['body'].apply(nltk.tokenize.casual.reduce_lengthening)

# remove specific chars or character groups RTs, colons,semicolons, hashtags and periods

tweets['body'] = tweets['body'].str.replace('RT|:|#|;|\.|\||\[|\]|\(|\)|@|\\|\/',' ').str.replace('&amp;|&amp','and').str.replace('&gt','').str.replace('\n',' ').str.strip()

# cleanup duplicates - again
tweets.drop_duplicates(['body'], keep='last', inplace=True)
tweets.reset_index(drop=True, inplace=True)

# remove non-english entries

# manually skip wrongly identified english tweets that would oth be removed
skiptweets = [35,55,69,163,198,207,216,227,240,244,249,323,330,334,698,701,702,814,818,820,984,1156,1200,1325,1410,1451,1502,1503,1677,1691,1692,1714,1720,1727,1744,1757,1875,1953,1971,2014,2043,2081,2090,2172]
dtweets = []

# first pass using langdetect
for i in range(0,len(tweets)):
    try:
        tweetlang = detect(tweets['body'][i])
    except lang_detect_exception.LangDetectException:
        dtweets += [i]
    else:
        if not tweetlang == 'en':
            if i not in skiptweets:
                print(i,tweets['body'][i])
                dtweets += [i]

tweets.drop(tweets.index[dtweets], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# second pass using non-ascii char detection
skiptweets = [98,304,312,314,319,327,454,640,1353,1402,1403,1589,1605,1736,1858]
dtweets = []

for i in range(0,len(tweets)):
    perc_ascii = (len(tweets['body'][i].encode()) - len(tweets['body'][i])) / len(tweets['body'][i].encode())
    if perc_ascii > 0.08:
        if i not in skiptweets:
            print(i,tweets['body'][i])
            dtweets += [i]

tweets.drop(tweets.index[dtweets], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# possibly use https://github.com/seatgeek/fuzzywuzzy to drop further similar tweets here

# remove remaining non-ascii chars from tweets

tweets['body'] = tweets['body'].str.encode("ascii", errors="ignore").str.decode("ascii")

# save cleaned data set

tweets.to_json('./cleanData/tweets_gen_clean.json')
tweets.to_csv('./cleanData/tweets_gen_clean.csv')

# remove title terms

#for i in len(tweets):
#    for j in list(set(ast.literal_eval(tweets['cites_papers'][i])))
#        response = requests.get('https://api.altmetric.com/v1/id/'+list(set(j)))
#        json_data = json.loads(response.text)
#        print
