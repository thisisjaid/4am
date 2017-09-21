# 4am
This is a research project for the 4:AM conference that investigates whether gender studies research attracts more trolling (or negative sentiment) on Twitter than other types of research (paleontology (STEM) and cultural studies (other Humanities)).

## Requirements
* Python3
* NLTK (pip install nltk)
* Matplotlib (pip install matplotlib) for graphs
* langdetect (pip install langdetect) for language detection
* Pandas (pip install pandas) for data manipulation
* SentiStrengthCom.jar - this is the commercial JAVA version of SentiStrength and is not included due to licensing but can be obtained free for research purposes from http://sentistrength.wlv.ac.uk/ (SentiStrengthData which is the word term data sets for SentiStrength are free for general use and _are_ included)

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

## Acknowledgements
Thanks to @konieczkow for helping with data export on short notice.
