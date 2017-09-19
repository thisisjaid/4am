import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import subprocess
import shlex
import sys

# Print most common terms in dataset

#pd.options.display.max_rows = 100
#print('Most common terms in data set (top 100)',pd.Series(' '.join(tweets['body']).lower().split()).value_counts()[:100])

# calculate average sentiment with nltk

tweets = pd.read_json('./cleanData/tweets_paleo_clean_notitles.json')

sid = SentimentIntensityAnalyzer()

nltk_com = []
nltk_pos = []
nltk_neg = []
nltk_neu = []

for i in range(0,len(tweets)):
    pscores = sid.polarity_scores(tweets['body'][i])
    nltk_com.append(pscores['compound'])
    nltk_pos.append(pscores['pos'])
    nltk_neg.append(pscores['neg'])
    nltk_neu.append(pscores['neu'])

tweets['nltk_com'] = nltk_com
tweets['nltk_pos'] = nltk_pos
tweets['nltk_neg'] = nltk_neg
tweets['nltk_neu'] = nltk_neu

print('NLTK Compound average:',tweets['nltk_com'].mean())
print('NLTK positive average:',tweets['nltk_pos'].mean())
print('NLTK negative average:',tweets['nltk_neg'].mean())
print('NLTK neutral average:',tweets['nltk_neu'].mean())

# Write NLTK info to a stats file

stats = open('./analysis/nltk_scores_paleo', 'w+')
print('NLTK Compound average:',tweets['nltk_com'].mean(),file=stats)
print('NLTK positive average:',tweets['nltk_pos'].mean(),file=stats)
print('NLTK negative average:',tweets['nltk_neg'].mean(),file=stats)
print('NLTK neutral average:',tweets['nltk_neu'].mean(),file=stats)
stats.close()

# Run through SentiStrength
def RateSentiment(sentiString):
    p = subprocess.Popen(shlex.split("java -jar SentiStrengthCom.jar stdin scale sentidata SentiStrengthData/"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    return stdout_text

ss_com = []
ss_pos = []
ss_neg = []

for i in range(0,len(tweets)):
    ssrating = RateSentiment(tweets['body'][i]).split()
    ss_com.append(int(ssrating[2]))
    ss_pos.append(int(ssrating[0]))
    ss_neg.append(int(ssrating[1]))
    print(i/len(tweets)*100," percent complete", end='\r')
    sys.stdout.flush()

tweets['ss_com'] = ss_com
tweets['ss_pos'] = ss_pos
tweets['ss_neg'] = ss_neg

print('SentiStrength Compound average:',tweets['ss_com'].mean())
print('SentiStrength positive average:',tweets['ss_pos'].mean())
print('SentiStrength negative average:',tweets['ss_neg'].mean())

# Write SS info to a stats file

stats = open('./analysis/ss_scores_paleo', 'w+')
print('SentiStrength Compound average:',tweets['ss_com'].mean(),file=stats)
print('SentiStrength positive average:',tweets['ss_pos'].mean(),file=stats)
print('SentiStrength negative average:',tweets['ss_neg'].mean(),file=stats)
stats.close()

plt.figure()
tweets['nltk_com'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology NLTK Compound Score Distribution')
plt.savefig("graphs/paleo_nltk_com.svg")

plt.clf()
tweets['nltk_pos'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology NLTK Positive Score Distribution')
plt.savefig("graphs/paleo_nltk_pos.svg")

plt.clf()
tweets['nltk_neg'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology NLTK Negative Score Distribution')
plt.savefig("graphs/paleo_nltk_neg.svg")

plt.clf()
tweets['nltk_neu'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology NLTK Neutral Score Distribution')
plt.savefig("graphs/paleo_nltk_neu.svg")

plt.clf()
tweets['ss_com'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology SentiStrength Compound Score Distribution')
plt.savefig("graphs/paleo_ss_com.svg")

plt.clf()
tweets['ss_pos'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology SentiStrength Positive Score Distribution')
plt.savefig("graphs/paleo_ss_pos.svg")

plt.clf()
tweets['ss_neg'].hist(color='k', alpha=0.5, bins=50)
plt.title('Paleontology SentiStrength Negative Score Distribution')
plt.savefig("graphs/paleo_ss_neg.svg")
