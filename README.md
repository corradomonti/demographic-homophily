# Evidence of Demographic rather than Ideological Segregation in News Discussion on Reddit

This repository contains code and data to reproduce our results from ***"Evidence of Demographic rather than Ideological Segregation in News Discussion on Reddit"*** by Corrado Monti, Jacopo D’Ignazi, Michele Starnini and Gianmarco De Francisci Morales, published at [ACM Web Conference 2023 (WWW'23)](https://www2023.thewebconf.org). If you use the provided data or code, we would appreciate a citation to the paper:

```
@inproceedings{monti2023evidence,
  title={Evidence of Demographic rather than Ideological Segregation in News Discussion on Reddit},
  author={Monti, Corrado and D'Ignazi, Jacopo and Starnini, Michele and De Francisci Morales, Gianmarco},
  booktitle={Proceedings of the ACM Web Conference 2023 (WWW ’23)},
  year={2023}
}
```

Here you will find (i) the (anonymized) Reddit dataset we presented in the paper and (ii) code to reproduce our experiments.

## Reddit Demographic Homophily Data Set

[You can download our anonymized Reddit `r/news` data set from here](https://github.com/corradomonti/demographic-homophily/releases/tag/dataset).

For every year 2016 to 2020 (included), the data set contains these three CSV files.
Each username is consistently replaced with an anonymized string.

- `YEAR_news_authors.csv`: for each Reddit users included in the analysis (non-bots users with at least 25 messages on r/news and at least one submission in 5 different subreddits in that year), this file reports their anonymized username and their score on the age, gender, partisan, and affluence axes. Scores are quantile-normalized, so that i.e. a score of 0.25 indicates the 25th percentile. The axes respectively correspond to probability of being young (low) or old (high), male or female, poor or rich, and left-leaning or right-leaning.

- `YEAR_news_graph.csv`: each line corresponds to a comment on r/news in that year. The file lists an anonymized id for the submission under which the comment happens, the author of the comment, the author of the parent comment to which this comment is replying to, and the sentiment of the text of the interaction. This can be seen as a weighted graph among users.

- `YEAR_news_submissions.csv`: each line corresponds to a submission on r/news, including the anonymized id of the submission, username of its author, total number of comments received, and the topic of the submission.

See the paper for more details about how we extracted this information.
The total number of considered users and comments per year is

| Year     | 2016    | 2017    | 2018    | 2019   | 2020    |
|----------|---------|---------|---------|--------|---------|
| N. nodes | 27976   | 34060   | 31997   | 21225  | 29045   |
| N. edges | 1166076 | 1390243 | 1221779 | 793569 | 1067614 |

## Reproducibility

In order to reproduce our experiments, we provide all our code to generate the analysis and the notebooks to generate the plot from the data set. In particular, you have to:

- [download the data set](https://github.com/corradomonti/demographic-homophily/releases/tag/dataset) and unzip its content in `data/reddit/anonymized`;
- create the [Conda environment](https://github.com/corradomonti/demographic-homophily/blob/main/environment.yml): `conda env create -f environment.yml`
- run the [regression model analysis](https://github.com/corradomonti/demographic-homophily/blob/main/src/main.py): `python main.py`
- this will (re)generate [the table summarizing the results of the model: `data/results/sd-model-regression-results.csv`](https://github.com/corradomonti/demographic-homophily/blob/main/data/results/sd-model-regression-results.csv)
- run the regression model analysis with topics: `python main.py --topics`
- this will (re)generate the two tables summarizing the results of the model with topics: the [interaction matrix among demographic classes](https://github.com/corradomonti/demographic-homophily/blob/main/data/results/sd-topics-model-regression-results.csv) and the [demographic class-topic interaction matrix](https://github.com/corradomonti/demographic-homophily/blob/main/data/results/sd-topics-model-regression-topic-results.csv).
- You can generate the plots from the paper using the [provided notebooks](https://github.com/corradomonti/demographic-homophily/tree/main/notebook).

For further information or needed data, please contact me: `me@corradomonti.com`.
