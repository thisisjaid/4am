# Poster Draft

## Introduction

Among the uninitiated, more metrics for research is often interpreted to be positive. After all, if more people are talking about a study, isn’t that a good thing? 

We suggest that more metrics are not always good, especially for certain humanities disciplines. In this study, we examine the effects of disciplinarity upon altmetrics for humanities research, comparing data for gender studies research published in the field's top ten (by SciMago Journal Rank) journals with articles published in the top ten cultural studies (humanities) and paleontology (STEM) journals. Specifically, we test our assumption that gender studies research is subject to more negative discussions than other research areas. 

## Methods

1. Extracted top ten journals and their ISSNs from SciMago Journal Rank
2. Added ISSNs to Altmetric Explorer to find all content with Altmetric attention published in those journals in 2016
3. Analyzed journal-level attention by discipline
4. Extracted full-text Twitter mentions directly from Altmetric database for sentiment analysis
5. Cleaned Twitter dataset for each discipline by removing article titles, usernames, and duplicate tweets from tweet full-text [ALEX, ANYTHING ELSE HERE?]
6. Analyzed each Twitter dataset using the Natural Language Tool Kit's built-in sentiment analysis tools
7. Analyzed each Twitter dataset again using SentiStrength

High degree of duplicate tweets - start with 6238 tweets, 3063 left after first duplicate pass, with 3175 (51%) removed; after removing URLs, handles and retweet tags, a further 865 (13%) duplicates are removed; a total of 4040 (64%) duplicate tweets were removed from the starting dataset
High degree of retweeting - out of initial 6238 tweet dataset, 3976 (63.7%) were retweets


## Differences in platforms for discussion 

_note the online platforms used most to discuss and share research in particular humanities disciplines_

## Differences in sentiment, by discipline

_maybe a side-by-side dataviz here for each disciplines_

## Sentiment Scores (NLTK & SentiStrength)

We analyzed each field's Twitter mentions using both the Python Natural Language Tool Kit and SentiStrength software. Here's a statement summariznig the most important conclusion. 

...

## Key findings

We then discuss the implications of our findings for altmetrics services and the evaluators who use them.

## Challenges

we looked at de-deduped gender tweet set, didn’t find any meaningful differences with stats - but it's possible that our approach means the averages have been affected.

non-identical but similar tweets (hard to disambiguate automatically, do you keep longer or shorter tweet, would have to test on error rate, etc)

Example:

"‘Washing men's feet’: gender, care and migration in Albania during &amp; after communism, J. Vullnetari &amp; R. King https://t.co/gTPhhNoZZu"

"‘Washing men's feet’: gender, care and migration in Albania during and after communism : http://t.co/a4BnorP48A"

We did not manually code tweets (like Freidrich et al)

## References

NLTK reference

SentiStrength reference

## Acknowledgements

Many thanks to Dr. Mike Thelwall for granting us access to SentiStrength software, and Maciej Gajewski for his assistance in data retrieval. Thanks also to Jean Liu for this poster’s visual precedent.


