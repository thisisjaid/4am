import pandas as pd
import json
import requests
import re
import numpy as np
from nltk.tokenize import TweetTokenizer

########################  Gender studies ##########################

def title_clean(dataset):

    tweets = pd.read_json('./cleanData/tweets_'+dataset+'_clean.json')

    # set column type to string
    tweets['body'] = tweets['body'].astype(str)
    tweets['cites_papers'] = tweets['cites_papers'].astype(str)

    # remove title terms

    tchanged = 0

    for i in range(0,len(tweets)):
        for j in eval(tweets['cites_papers'][i]):
            response = requests.get('https://api.altmetric.com/v1/id/'+str(j))
            json_data = json.loads(response.text)
            if tweets['body'][i] != tweets['body'][i].replace(json_data['title'],''):
                tweets['body'][i] = tweets['body'][i].replace(json_data['title'],'')
                tchanged += 1

    print("Total records changed",tchanged)

    # clean up barren tweets (empty or very short)
    dtweets = []

    for i in range(0,len(tweets)):
        if len(tweets['body'][i]) <= 3:
            dtweets += [i]

    tweets.drop(tweets.index[dtweets], inplace=True)
    tweets.reset_index(drop=True, inplace=True)

    # save cleaned data set

    tweets.to_json('./cleanData/tweets_'+dataset+'_clean_notitles.json')
    tweets.to_csv('./cleanData/tweets_'+dataset+'_clean_notitles.csv')

title_clean('gen')
title_clean('paleo')
title_clean('cult')
