import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Print most common terms in dataset

#pd.options.display.max_rows = 100
#print('Most common terms in data set (top 100)',pd.Series(' '.join(tweets['body']).lower().split()).value_counts()[:100])

# calculate average sentiment with nltk

tweets = pd.read_json('./cleanData/tweets_paleo_clean.json')

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

stats = open('./analysis/nltk_scores_gen', 'w+')
print('NLTK Compound average:',sentiavg,file=stats)
print('NLTK positive average:',sentiposavg,file=stats)
print('NLTK negative average:',sentinegavg,file=stats)
print('NLTK neutral average:',sentineuavg,file=stats)
stats.close()
