import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
import shlex
import sys

def RateSentiment(sentiString):
    p = subprocess.Popen(shlex.split('java -jar SentiStrengthCom.jar stdin scale sentidata SentiStrengthData/'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b = bytes(sentiString.replace(' ','+'), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode('utf-8')
    return stdout_text

# set of specific Altmetric paper IDs we want to pull from data set
# to analyse tweet scores for
specific = [5110616,6050273,7360918,8175741,6147115,12469790,5980250,6164135,6524203,13260383]

# load data set
tweets = pd.read_json('./cleanData/tweets_gen_clean_notitles.json')

# initialize NLTK VADER analyzer
sid = SentimentIntensityAnalyzer()

# create arrays to store scores in for each class
nltk_com = []
nltk_pos = []
nltk_neg = []
nltk_neu = []
numspecific = 0

# calculate and store scores
for i in range(0,len(tweets)):
    if pd.Series(eval(tweets['cites_papers'][i])).isin(specific).any():    
        pscores = sid.polarity_scores(tweets['body'][i])
        nltk_com.append(pscores['compound'])
        nltk_pos.append(pscores['pos'])
        nltk_neg.append(pscores['neg'])
        nltk_neu.append(pscores['neu'])
        numspecific += 1

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

stats = open('./analysis/nltk_scores_specific_gen', 'w+')
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
numspecific = 0

# calculate and store scores
for i in range(0,len(tweets)):
    if pd.Series(eval(tweets['cites_papers'][i])).isin(specific)any():    
        ssrating = RateSentiment(tweets['body'][i]).split()
        ss_com.append(int(ssrating[2]))
        ss_pos.append(int(ssrating[0]))
        ss_neg.append(int(ssrating[1]))
        print(i/len(tweets)*100,' percent complete', end='\r')
        sys.stdout.flush()
        numspecific += 1

# create new columns
tweets['ss_com'] = ss_com
tweets['ss_pos'] = ss_pos
tweets['ss_neg'] = ss_neg

# calculate and print score means
print('SentiStrength Compound average:',tweets['ss_com'].mean())
print('SentiStrength positive average:',tweets['ss_pos'].mean())
print('SentiStrength negative average:',tweets['ss_neg'].mean())

# Write SS mean scores to a stats file
stats = open('./analysis/ss_scores_specific_gen', 'w+')
print('SentiStrength Compound average:',tweets['ss_com'].mean(),file=stats)
print('SentiStrength positive average:',tweets['ss_pos'].mean(),file=stats)
print('SentiStrength negative average:',tweets['ss_neg'].mean(),file=stats)
stats.close()
