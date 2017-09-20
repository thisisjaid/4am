import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
import shlex
import sys

# Print most common terms in dataset

#pd.options.display.max_rows = 100
#print('Most common terms in data set (top 100)',pd.Series(' '.join(tweets['body']).lower().split()).value_counts()[:100])

# calculate average sentiment with nltk

top10 = [5110616,6050273,7360918,8175741,6147115,12469790,5980250,6164135,6524203,13260383]

tweets = pd.read_json('./cleanData/tweets_gen_clean_notitles.json')

sid = SentimentIntensityAnalyzer()
sentisum = 0
sentipos = 0
sentineg = 0
sentineu = 0

numtop10 = 0

for i in range(0,len(tweets)):
    if pd.Series(eval(tweets['cites_papers'][i])).isin(top10).any():
        pscores = sid.polarity_scores(tweets['body'][i])
        sentisum += pscores['compound']
        sentipos += pscores['pos']
        sentineg += pscores['neg']
        sentineu += pscores['neu']
        numtop10 += 1

sentiavg = sentisum / numtop10
sentiposavg = sentipos / numtop10
sentinegavg = sentineg / numtop10
sentineuavg = sentineu / numtop10

print('NLTK Compound average:',sentiavg)
print('NLTK positive average:',sentiposavg)
print('NLTK negative average:',sentinegavg)
print('NLTK neutral average:',sentineuavg)

# Write NLTK info to a stats file

stats = open('./analysis/nltk_scores_gen_top10', 'w+')
print('NLTK Compound average:',sentiavg,file=stats)
print('NLTK positive average:',sentiposavg,file=stats)
print('NLTK negative average:',sentinegavg,file=stats)
print('NLTK neutral average:',sentineuavg,file=stats)
stats.close()


# Run through SentiStrength
def RateSentiment(sentiString):
    p = subprocess.Popen(shlex.split("java -jar SentiStrengthCom.jar stdin scale sentidata SentiStrengthData/"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    return stdout_text

ss_sum = 0
ss_pos = 0
ss_neg = 0

numtop10 = 0

for i in range(0,len(tweets)):
    if pd.Series(eval(tweets['cites_papers'][i])).isin(top10).any():
        ssrating = RateSentiment(tweets['body'][i]).split()
        ss_sum += int(ssrating[2])
        ss_pos += int(ssrating[0])
        ss_neg += int(ssrating[1])
        print(i/len(tweets)*100," percent complete", end='\r')
        sys.stdout.flush()
        numtop10 += 1

ss_avg = ss_sum / len(tweets)
ss_posavg = ss_pos / len(tweets)
ss_negavg = ss_neg / len(tweets)

print('SentiStrength Compound average:',ss_avg)
print('SentiStrength positive average:',ss_posavg)
print('SentiStrength negative average:',ss_negavg)

# Write SS info to a stats file

stats = open('./analysis/ss_scores_gen_top10', 'w+')
print('SentiStrength Compound average:',ss_avg,file=stats)
print('SentiStrength positive average:',ss_posavg,file=stats)
print('SentiStrength negative average:',ss_negavg,file=stats)
stats.close()
