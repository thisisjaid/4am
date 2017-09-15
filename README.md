# 4am
This is a research project for the 4:AM conference that's investigating whether gender studies research attracts more trolling on Twitter than other types of research (paleontology (STEM) and cultural studies (other Humanities)).

## CSV/JSON data files
These are exports of the full-text of tweets mentioning articles published in 2016 in the Top 10 gender studies, cultural studies, and paleontology journals (as determined by SciMago Journal Rank). These JSON files were exported directly from the Altmetric database on 8 September 2017. 

These files contain the following types of data:
* \_id: Altmetric's internal identifier for the "post" i.e. Twitter mention
* body: The full-text of the tweet that mentions a paper
* cites_papers: The paper(s) mentioned in the tweet

These files are kept in two folders: rawData, which contains the raw export pulled from Altmetric database; and cleanedData, which is the data that's been parsed.

## Analysis folder
Contains notes and drafts explaining various analyses run on our data, including
NLTK vs. Sentistrength sentiments.

## Acknowledgements
Thanks to @konieczkow for helping with data export on short notice.
