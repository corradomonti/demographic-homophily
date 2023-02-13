import numpy as np
import pandas as pd
import sklearn.utils
import statsmodels.api as sm
from tqdm import tqdm

import argparse
import logging
import sys

FEATURE2CLASSES = {
    'age': ['Young', 'Old'],
    'gender': ['Male', 'Female'],
    'affluence': ['Poor', 'Rich'],
    'partisan': ['Left', 'Right'],
}

CLASSES = [x for pair in FEATURE2CLASSES.values() for x in pair]

SELECTED_TOPICS = np.array([
    'BUSINESS', 'POLITICS', 'WORLDPOST', 'ENTERTAINMENT', 'HEALTHY LIVING', 'CRIME', 'SPORTS',
    'TRAVEL', 'GREEN', 'TECH', 'ARTS', 'MEDIA', 'STYLE', 'WEIRD NEWS', 'RELIGION'
])

def subsample_neg_pairs(graph_df):
    edges_set = set(map(tuple, graph_df.values))
    non_edges = []
    pbar = tqdm(total=len(graph_df))
    while len(non_edges) < len(graph_df):
        sample_size = min(2 ** 17, len(graph_df) - len(non_edges))  # 2**17 overflow
        column_samples = [np.random.choice(graph_df[c], sample_size) for c in graph_df.columns]
        new_non_edges = [sample for sample in zip(*column_samples) if sample not in edges_set]
        non_edges += new_non_edges
        pbar.update(len(new_non_edges))
    pbar.close()
    return pd.DataFrame(non_edges, columns=graph_df.columns)

def do_regression(year, dataset_path, with_topics=False, seed=42):
    np.random.seed(seed)
    logging.info('Reading input files for year %s...', year)
    author_features = pd.read_csv(dataset_path + f"{year}_news_authors.csv")
    graph_df = pd.read_csv(dataset_path + f"{year}_news_graph.csv", index_col=0)
    
    authors_with_features = set(author_features.author.unique())
    graph_df = graph_df[
        (graph_df.parent_author.isin(authors_with_features)) &
        (graph_df.author.isin(authors_with_features))
    ]
    graph_df = graph_df[(graph_df.author != graph_df.parent_author)]
    
    if with_topics:
        submissions_df = pd.read_csv(dataset_path + f"{year}_news_submissions.csv")
        graph_df = pd.merge(graph_df, submissions_df[['submission_id', 'topic']],
                   left_on='submission_id', right_on='submission_id')
        graph_df = graph_df[graph_df.topic.isin(SELECTED_TOPICS)]
        graph_df = graph_df[['parent_author', 'author', 'topic']].copy()
    else:
        graph_df = graph_df[['parent_author', 'author']].copy()
    
    logging.info('Preparing regression table...')
    for feature, (class_low, class_hi) in FEATURE2CLASSES.items():
        author_features[class_low] = author_features[feature] <= 0.25
        author_features[class_hi] = author_features[feature] >= 0.75
        del author_features[feature]
    
    logging.info('Computing negative pairs subsample...')
    non_edges_df = subsample_neg_pairs(graph_df)
    non_edges_df['is_link'] = 0.
    graph_df['is_link'] = 1.
    
    logging.info('Crossing features with selected pairs...')
    regression_df = sklearn.utils.shuffle(
        pd.concat([graph_df, non_edges_df])
        .merge(author_features, left_on='author', right_on='author')
        .merge(author_features, left_on='parent_author', right_on='author',
               suffixes=('_child', '_parent'))
    )
    
    logging.info('Applying outer product kernel...')
    regression_variables = [(
            f"{c1}_child*{c2}_parent", 
            regression_df[f"{c1}_child"] & regression_df[f"{c2}_parent"]
        ) for c1 in CLASSES for c2 in CLASSES
    ]
    
    if with_topics:
        regression_variables += [(
            f"{c}*{t}", (
                (regression_df[f"{c}_child"] | regression_df[f"{c}_parent"]) &
                (regression_df.topic == t)
            )
        ) for c in CLASSES for t in SELECTED_TOPICS]
    
    variable_names = [name for name, value in regression_variables]
    X = np.vstack([value for name, value in regression_variables]).astype(float).T
    
    logging.info(f'Performing regression (independent variables matrix has shape {X.shape})...')
    logreg = sm.Logit(np.array(regression_df.is_link.values),
                      sm.add_constant(X, has_constant='add'),
                      missing='raise')
    logreg_res = logreg.fit(maxiter=1000, method='lbfgs', disp=True)
    return logreg_res, variable_names

def do_results_table(logreg_res, variable_names, year, with_topics=False):   
    results_rows = []
    conf_int = logreg_res.conf_int(alpha=0.05)
    for a in CLASSES:
        for b in (SELECTED_TOPICS if with_topics else CLASSES):
            idx = variable_names.index(
                f"{a}*{b}" if with_topics else f"{a}_child*{b}_parent"
            ) + 1 # for Intercept
            param = logreg_res.params[idx]
            lo, hi = conf_int[idx]
            signif = []
            for alpha in (0.05, 0.01, 0.001):
                lo, hi = logreg_res.conf_int(alpha=alpha)[idx]
                signif.append((lo * hi) > 0)
            results_rows += [[a, b, year, param, (param - lo), (hi - param)] + signif]
    
    results_columns = ["class", "topic"] if with_topics else ["class_child", "class_parent"]
    results_columns += ['year', 'param', 'down_err', 'up_err', 'signif', 'signif01', 'signif001']
    return pd.DataFrame(results_rows, columns=results_columns)

def main(with_topics=False, dataset_path="../data/reddit/anonymized/"):
    results_dfs = []
    t_results_dfs = [] if with_topics else None
    for year in [2016, 2017, 2018, 2019, 2020]:
        logreg_res, variable_names = do_regression(year, dataset_path, with_topics=with_topics)
        results_dfs += [do_results_table(logreg_res, variable_names, year, with_topics=False)]
        if with_topics:
            t_results_dfs += [do_results_table(logreg_res, variable_names, year, with_topics=True)]
    
    output_path = ("../data/results/" + ('sd-topics' if with_topics else 'sd') + 
        "-model-regression-results.csv")
    pd.concat(results_dfs).to_csv(output_path, index=False)
    logging.info("Saved results to %s", output_path)
    
    if with_topics:
        t_output_path = ("../data/results/sd-topics-model-regression-topic-results.csv")
        pd.concat(t_results_dfs).to_csv(t_output_path, index=False)
        logging.info("Saved topic results to %s", t_output_path)
    
    logging.info("Done. ✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform regression analysis with selected model.')
    parser.add_argument('--dataset', help='Dataset path', default="../data/reddit/anonymized/")
    parser.add_argument('--topics', action='store_true')
    args = parser.parse_args()
    
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    main(with_topics=args.topics, dataset_path=args.dataset)
