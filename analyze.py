import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import subprocess
import shlex
import sys

# definitely not the fastest way to call SentiStrength - to be reworked
def RateSentiment(sentiString):
    p = subprocess.Popen(shlex.split('java -jar SentiStrengthCom.jar stdin scale sentidata SentiStrengthData/'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b = bytes(sentiString.replace(' ','+'), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode('utf-8')
    return stdout_text

def analyze_tweets(setname):
    # load cleaned data
    tweets = pd.read_json('./cleanData/tweets_'+setname+'_clean_notitles.json')
    # run tweets through NLTK Vader Sentiment Analyzer
    sid = SentimentIntensityAnalyzer()
    # create arrays to store scores in for each class
    nltk_com = []
    nltk_pos = []
    nltk_neg = []
    nltk_neu = []
    # calculate and store scores
    for i in range(0,len(tweets)):
        pscores = sid.polarity_scores(tweets['body'][i])
        nltk_com.append(pscores['compound'])
        nltk_pos.append(pscores['pos'])
        nltk_neg.append(pscores['neg'])
        nltk_neu.append(pscores['neu'])
    # add dataframe columns
    tweets['nltk_com'] = nltk_com
    tweets['nltk_pos'] = nltk_pos
    tweets['nltk_neg'] = nltk_neg
    tweets['nltk_neu'] = nltk_neu
    # calculate and print means
    print('NLTK Compound average:',tweets['nltk_com'].mean())
    print('NLTK positive average:',tweets['nltk_pos'].mean())
    print('NLTK negative average:',tweets['nltk_neg'].mean())
    print('NLTK neutral average:',tweets['nltk_neu'].mean())
    # write NLTK mean scores to a stats file
    stats = open('./analysis/nltk_scores_'+setname, 'w+')
    print('NLTK Compound average:',tweets['nltk_com'].mean(),file=stats)
    print('NLTK positive average:',tweets['nltk_pos'].mean(),file=stats)
    print('NLTK negative average:',tweets['nltk_neg'].mean(),file=stats)
    print('NLTK neutral average:',tweets['nltk_neu'].mean(),file=stats)
    stats.close()
    # Run through SentiStrength
    # create score storage arrays
    ss_com = []
    ss_pos = []
    ss_neg = []
    # calculate and store scores
    for i in range(0,len(tweets)):
        ssrating = RateSentiment(tweets['body'][i]).split()
        ss_com.append(int(ssrating[2]))
        ss_pos.append(int(ssrating[0]))
        ss_neg.append(int(ssrating[1]))
        print(i/len(tweets)*100,' percent complete', end='\r')
        sys.stdout.flush()
    # create new columns
    tweets['ss_com'] = ss_com
    tweets['ss_pos'] = ss_pos
    tweets['ss_neg'] = ss_neg
    # calculate and print score means
    print('SentiStrength Compound average:',tweets['ss_com'].mean())
    print('SentiStrength positive average:',tweets['ss_pos'].mean())
    print('SentiStrength negative average:',tweets['ss_neg'].mean())
    # Write SS mean scores to a stats file
    stats = open('./analysis/ss_scores_'+setname, 'w+')
    print('SentiStrength Compound average:',tweets['ss_com'].mean(),file=stats)
    print('SentiStrength positive average:',tweets['ss_pos'].mean(),file=stats)
    print('SentiStrength negative average:',tweets['ss_neg'].mean(),file=stats)
    stats.close()
    # plot and save graphs
    # nltk graphs
    plt.figure()
    tweets['nltk_com'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' NLTK Compound Score Distribution')
    plt.savefig('graphs/'+setname+'_nltk_com.svg')
    plt.clf()
    tweets['nltk_pos'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' NLTK Positive Score Distribution')
    plt.savefig('graphs/'+setname+'_nltk_pos.svg')
    plt.clf()
    tweets['nltk_neg'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' NLTK Negative Score Distribution')
    plt.savefig('graphs/'+setname+'_nltk_neg.svg')
    plt.clf()
    tweets['nltk_neu'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' NLTK Neutral Score Distribution')
    plt.savefig('graphs/'+setname+'_nltk_neu.svg')
    # SS graphs
    plt.clf()
    plt.xlim(-4, 4) # force x limits to keep graph aspect
    tweets['ss_com'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' SentiStrength Compound Score Distribution')
    plt.savefig('graphs/'+setname+'_ss_com.svg')
    plt.clf()
    tweets['ss_pos'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' SentiStrength Positive Score Distribution')
    plt.savefig('graphs/'+setname+'_ss_pos.svg')
    plt.clf()
    tweets['ss_neg'].hist(color='k', alpha=0.5, bins=50)
    plt.title(graph_titles[setname]+' SentiStrength Negative Score Distribution')
    plt.savefig('graphs/'+setname+'_ss_neg.svg')


# define graph titles array
graph_titles = {
    'gen': 'Gender Studies',
    'cult': 'Cultural Studies',
    'paleo': 'Paleontology'
}

# run analysis
analyze_tweets('gen')
analyze_tweets('cult')
analyze_tweets('paleo')
