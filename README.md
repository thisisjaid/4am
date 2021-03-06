# 4am
This is a research project for the 4:AM conference that investigates whether gender studies research attracts more trolling (or negative sentiment) on Twitter than other types of research (paleontology (STEM) and cultural studies (other Humanities)).

## Requirements
* Python3
* NLTK (pip install nltk) - processing and VADER sentiment analysys tools
* Matplotlib (pip install matplotlib) for graphs
* langdetect (pip install langdetect) for language detection
* Pandas (pip install pandas) for data manipulation
* SentiStrengthCom.jar - must be placed in the root of the repository - this is the commercial JAVA version of SentiStrength and is not included due to licensing but can be obtained free for research purposes from http://sentistrength.wlv.ac.uk/ (SentiStrengthData which is the word term data sets for SentiStrength are free for general use and _are_  included) 

## Files
* clean_data.py - main data cleanup script, see contents for details on cleanup operations performed
* analyze.py - main data analysis script
* analyze_specific.py - customised data analysis script for scoring tweets for specific subsets of papers identified by Altmetric internal ID

## CSV/JSON data files
These are exports of the full-text of tweets mentioning articles published in 2016 in the Top 10 gender studies, cultural studies, and paleontology journals (as determined by SciMago Journal Rank). These JSON files were exported directly from the Altmetric database on 17 September 2017. 

These files contain the following types of data:
* \_id: Altmetric's internal identifier for the "post" i.e. Twitter mention
* body: The full-text of the tweet that mentions a paper
* cites_papers: The Altmetric internal identifier for the paper(s) mentioned in the tweet

These files are kept in two folders: rawData, which contains the raw export pulled from Altmetric database; and cleanedData, which is the data that's been parsed.

## Analysis folder
Contains score outputs for the two scoring algorithms used for each data set.

## Graphs folder
Contains graphs of score distribution for each scoring algorithm and each data set.

## Licenses and copyright
All of the code and sample data in this repository is released under GPLv3 General License, with the exception of the SentiStrengthData directory which although free for use, remains the intellectual property of the SentiStrength authors (see http://sentistrength.wlv.ac.uk/ for details)

## Acknowledgements
Many thanks to Dr. Mike Thelwall for granting us access to SentiStrength software.
Thanks to @konieczkow for helping with data export on short notice.

VADER sentiment analysis tools
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014
