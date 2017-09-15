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
DetectorFactory.seed = 0

tweets = pd.read_json('./gendertweets-august.json',orient='records',lines=True) # Be sure to also update the export file so you don't overwrite those parsed tweets

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

tweets['body'] = tweets['body'].str.replace('RT|:|#|;|\.','').str.replace('&amp;|&amp','and').str.replace('\n',' ').str.strip()

# cleanup duplicates - again
tweets.drop_duplicates(['body'], keep='last', inplace=True)
tweets.reset_index(drop=True, inplace=True)

# remove non-english entries

# manually skip wrongly identified english tweets that would oth be removed
skiptweets = [35, 55, 69, 165, 200, 210, 218, 229, 242, 246, 251, 326, 697, 700, 701, 756, 813, 817, 819, 983, 1158, 1202, 1411, 1453, 1505, 1506, 1719, 1756, 1949, 2036, 2074, 2083, 2167]
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
                dtweets += [i]

tweets.drop(tweets.index[dtweets], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# second pass using non-ascii char detection
skiptweets = [98, 314, 319, 444, 1345, 1395, 1396, 1575]
dtweets = []

for i in range(0,len(tweets)):
    perc_ascii = (len(tweets['body'][i].encode()) - len(tweets['body'][i])) / len(tweets['body'][i].encode())
    if perc_ascii > 0.08:
        if i not in skiptweets:
            dtweets += [i]

tweets.drop(tweets.index[dtweets], inplace=True)
tweets.reset_index(drop=True, inplace=True)

#manually remove remaining non-english stragglers
mdtweets = [1677,1678]
tweets.drop(tweets.index[mdtweets], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# possibly use https://github.com/seatgeek/fuzzywuzzy to drop further similar tweets here

# remove remaining non-ascii chars from tweets

tweets['body'] = tweets['body'].str.encode("ascii", errors="ignore").str.decode("ascii")

# Write unique(ish) tweets to new text file so can be analyzed using SentiStrength
np.savetxt('./gendertweets-aug.txt', tweets.values, fmt='%s', newline='\n')

# calculate averate sentiment with nltk

sid = SentimentIntensityAnalyzer()
sentisum = 0
sentipos = 0
sentineg = 0
sentineu = 0

for i in range(0,len(tweets)):
    pscores = sid.polarity_scores(tweets['body'][i])
    sentisum += pscores['compound']
    sentipos += pscores['pos']
    sentineg += pscores['neg']
    sentineu += pscores['neu']

sentiavg = sentisum / len(tweets)
sentiposavg = sentipos / len(tweets)
sentinegavg = sentineg / len(tweets)
sentineuavg = sentineu / len(tweets)

print('NLTK Compound average:',sentiavg)
print('NLTK positive average:',sentiposavg)
print('NLTK negative average:',sentinegavg)
print('NLTK neutral average:',sentineuavg)

# Write NLTK info to a stats file

#pd.options.display.max_rows = 100
#print('Most common terms in data set (top 100)',pd.Series(' '.join(tweets['body']).lower().split()).value_counts()[:100])


# remove title terms

#for i in len(tweets):
#    for j in list(set(ast.literal_eval(tweets['cites_papers'][i])))
#        response = requests.get('https://api.altmetric.com/v1/id/'+list(set(j)))
#        json_data = json.loads(response.text)
#        print
