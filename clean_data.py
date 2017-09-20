import pandas as pd
import json
import requests
import re
import json
import requests
from nltk import tokenize
from langdetect import lang_detect_exception
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0 # prevent langdetect from producing spurious results between runs

def title_clean(dataset):
    tchanged = 0

    for i in range(0,len(dataset)):
        # parse the papers cited in each tweet
        for j in eval(dataset['cites_papers'][i]):
            # grab the citation title from Altmetric API
            response = requests.get('https://api.altmetric.com/v1/id/'+str(j))
            json_data = json.loads(response.text)
            # only attempt to replace the title if it's actually matched
            if dataset['body'][i] != dataset['body'][i].replace(json_data['title'],''):
                dataset['body'][i] = dataset['body'][i].replace(json_data['title'],'')
                tchanged += 1

    print("Total records changed",tchanged)

    # clean up barren tweets (empty or very short)
    dataset = dataset[dataset['body'].map(len) > 3]
    dataset.reset_index(drop=True, inplace=True)

    return dataset

def clean_non_english(dataset,skiptweets):
    dtweets = []
    for i in range(0,len(dataset)):
        try:
            tweetlang = detect(dataset['body'][i])
        except lang_detect_exception.LangDetectException:
            dtweets += [i]
        else:
            if not tweetlang == 'en':
                if i not in skiptweets:
                    print(i,dataset['body'][i])
                    dtweets += [i]
    dataset.drop(dataset.index[dtweets], inplace=True)
    dataset.reset_index(drop=True, inplace=True)

def clean_non_ascii(dataset,skiptweets):
    dtweets = []
    for i in range(0,len(dataset)):
        perc_ascii = (len(dataset['body'][i].encode()) - len(dataset['body'][i])) / len(dataset['body'][i].encode())
        if perc_ascii > 0.08:
            if i not in skiptweets:
                print(i,dataset['body'][i])
                dtweets += [i]
    dataset.drop(dataset.index[dtweets], inplace=True)
    dataset.reset_index(drop=True, inplace=True)

def clean_tweets(setname,skiptweets_ne,skiptweets_na):

    tweets = pd.read_json('./rawData/tweets_'+setname+'.json',orient='records',lines=True)

    # drop id column which we don't actually need
    tweets.drop('_id', axis=1, inplace=True)

    # set column type to string
    tweets['body'] = tweets['body'].astype(str)
    tweets['cites_papers'] = tweets['cites_papers'].astype(str)

    # make columns format list-like so we can parse with eval into arrays later on
    tweets['cites_papers'] = tweets['cites_papers'].str.replace('{\'\$numberLong\':','').str.replace('\'','').str.replace('}','').str.replace(' ','')

    # cleanup duplicates
    tweets.drop_duplicates(['body'], keep='last', inplace=True)
    tweets.reset_index(drop=True, inplace=True)

    # remove URLS and mistyped URLs missing semicolons as well as http(s) stragglers
    tweets['body'].replace(re.compile(r"http.?:?//[^\s]+[\s]?"), "", inplace=True)
    tweets['body'].replace(re.compile(r"http.?:?/?/?"), "", inplace=True)

    # remove handles
    tweets['body'] = tweets['body'].apply(tokenize.casual.remove_handles)

    # reduce characters that repeat more than 3 times to 3 instances
    tweets['body'] = tweets['body'].apply(tokenize.casual.reduce_lengthening)

    # remove/replace specific chars or character groups RTs, colons,semicolons, hashtags and periods and extraneous space
    tweets['body'] = tweets['body'].str.replace('RT|:|#|;|\.|\||\[|\]|\(|\)|@|\\|\/',' ').str.replace('&amp;|&amp','and').str.replace('&gt','').str.replace('\n',' ').str.strip()

    # cleanup duplicates again
    tweets.drop_duplicates(['body'], keep='last', inplace=True)
    tweets.reset_index(drop=True, inplace=True)

    # use language detection to remove non-english entries
    clean_non_english(tweets,skiptweets_ne)

    # use non-ascii char detection to remove any remaining un-parseable tweets
    clean_non_ascii(tweets,skiptweets_na)

    # remove remaining non-ascii chars from tweets
    tweets['body'] = tweets['body'].str.encode("ascii", errors="ignore").str.decode("ascii")

    # remove titles
    tweets = title_clean(tweets)

    # save cleaned data set
    tweets.to_json('./cleanData/tweets_'+setname+'_clean_notitles.json')
    tweets.to_csv('./cleanData/tweets_'+setname+'_clean_notitles.csv')

# sets of tweets that are erroneously detected as non-english (manually selected)
skip_nonenglish = {
    'gen': [35,55,69,163,198,207,216,227,240,244,249,323,330,334,698,701,702,814,818,820,984,1156,1200,1325,1410,1451,1502,1503,1677,1691,1692,1714,1720,1727,1744,1757,1875,1953,1971,2014,2043,2081,2090,2172],
    'cult': [36,75,84,136,343,363,551,681,732,892,923,1035,1139,1169],
    'paleo': [43,61,125,154,226,357,504,592,620,622,707,772,922,955,956,1045,1134,1147,1300,1471,1687,1697,1698,2041,2071,2159,2198,2224,2350,2426,2595,2617,2620,2778,2814,2827,2826,2828,2830,2833,2837,2840,2848,2852,2863,2874,2883,2904,2504,2624,2654]
}

# sets of tweets that are erroneously detected as non-ascii (manually selected)
skip_nonascii = {
    'gen': [98,304,312,314,319,327,454,640,1353,1402,1403,1589,1605,1736,1858],
    'cult': [91,94,97,101,102,133],
    'paleo': [2529,2650,2686,2700]
}

# run cleanup
clean_tweets('gen',skip_nonenglish['gen'],skip_nonascii['gen'])
clean_tweets('cult',skip_nonenglish['cult'],skip_nonascii['cult'])
clean_tweets('paleo',skip_nonenglish['paleo'],skip_nonascii['paleo'])
