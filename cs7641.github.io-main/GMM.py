import os
import sys
import ast
import sklearn
import sklearn.mixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import random
import ipdb

# N = 33766
# D = 64

n_genres = 19
folds = 1
ff_oh = "pca_comp_data_input.npy"
# ff_literal = "final_data/final_processed_wo_empty_duplicates.csv"
ff_literal = "pca_comp_data_label.csv"
OUTF_metrics = "GMM_metrics.csv"


def create_ytrue_1D(d_path):
    data = pd.read_csv(d_path)
    ytrue = [ast.literal_eval(x)[0] for x in list(data["top_genre_index"])]
    return ytrue


def create_ytrue_OH(d_path):
    data = pd.read_csv(d_path)
    data = data.drop(columns=["Unnamed: 0"])

    ytrue = []
    for i, row in data.iterrows():
        c_idx = list(row).index(1.0)
        ytrue.append(c_idx)

    return ytrue


def load_data(oh_file, literal_file):
    data = np.load(oh_file)
    ytrue = create_ytrue_OH(literal_file)
    return data, ytrue


def data_split_5k(data, y_true, num_fold=5, k=0):
    divid_data = np.array_split(data, num_fold, axis=0)
    divid_y = np.array_split(y_true, num_fold, axis=0)

    test_n_val_data = np.array_split(divid_data[k], 2, axis=0)
    test_n_val_y = np.array_split(divid_y[k], 2, axis=0)

    test_data = test_n_val_data[0]
    test_y = test_n_val_y[0]
    val_data = test_n_val_data[1]
    val_y = test_n_val_y[1]

    del divid_data[k]
    del divid_y[k]
    train_data = np.concatenate(divid_data, axis=0)
    train_y = np.concatenate(divid_y, axis=0)

    return test_data, test_y, val_data, val_y, train_data, train_y


def most_common_y(preds, mapped):
    preds = pd.Series(preds)
    vcounts = preds.value_counts().rename_axis("labels").reset_index(name="counts")
    vcounts = vcounts.sort_values(by="counts", ascending=False)

    # NOTE: if a genre does not have all genres represented in preds,
    #       this may fail (failed for k=3, class imbalance issue)
    idx = 0
    label = vcounts["labels"][idx]
    while label in mapped:
        idx += 1
        label = vcounts["labels"][idx]

    return label


def map_components_to_labels(GMM, train_data, train_y):
    # will compute and return a dictionary that maps
    # class indexes in ytrue to component indexes in
    # GMM output
    ytrue = pd.Series(train_y)
    ytrue_set = ytrue.value_counts().rename_axis("labels").reset_index(name="counts")
    ytrue_set = ytrue_set.sort_values(by="counts", ascending=False)["labels"]
    map_dict = {}

    mapped = []
    for y in ytrue_set:
        idx_genre = [i for i, x in enumerate(train_y) if x == y]
        data_genre = train_data[idx_genre]
        gmm_preds = GMM.predict(data_genre)
        y_map = most_common_y(
            gmm_preds, mapped
        )  # the index from GMM prediction that appear most often from data_genre
        map_dict[str(y)] = y_map
        mapped.append(y_map)
        # ipdb.set_trace()

    return map_dict


def map_centers_to_labels(kmeans, train_data, train_y):
    pass
    # will compute and return a dictionary that maps
    # class indexes in ytrue to center indexes in
    # Kmeans output


def calc_metrics(y_true, y_pred, fold_num=0):
    acc_ = accuracy_score(y_true, y_pred)
    prec_ = precision_score(y_true, y_pred, average="weighted")
    rec_ = recall_score(y_true, y_pred, average="weighted")
    f1_ = f1_score(y_true, y_pred, average="weighted")

    data = {}
    data["k fold"] = [fold_num]
    data["precision score"] = [prec_]
    data["accuracy score"] = [acc_]
    data["recall score"] = [rec_]
    data["f1 score"] = [f1_]

    metrics = pd.DataFrame(data)
    print(metrics)
    return metrics


if __name__ == "__main__":

    data, ytrue = load_data(ff_oh, ff_literal)
    all_fold_metrics = []

    for k in range(folds):
        test_data, test_y, val_data, val_y, train_data, train_y = data_split_5k(
            data, ytrue, num_fold=5, k=k
        )

        GMM = sklearn.mixture.GaussianMixture(
            n_components=19, covariance_type="full", init_params="kmeans", random_state=0
        ).fit(train_data)

        y_map_dict = map_components_to_labels(GMM, train_data, train_y)
        val_y_translated = [y_map_dict[str(y)] for y in val_y]

        val_predictions = GMM.predict(val_data)
        val_pred_probs = GMM.predict_proba(val_data)

        metric_df = calc_metrics(val_y_translated, val_predictions, fold_num=k)
        all_fold_metrics.append(metric_df)

    all_metrics = pd.concat(all_fold_metrics)
    all_metrics.to_csv(OUTF_metrics, index=False)

    # saving df displaying class imbalance:
    final_data = pd.read_csv("final_data/final_processed_wo_empty_duplicates.csv")
    final_data = (
        final_data["top_genre_index"]
        .value_counts()
        .rename_axis("top_genres")
        .reset_index(name="counts")
    )
    final_data_genres = final_data.sort_values(by="counts", ascending=False)
    final_data_genres.to_csv("genre_class_imbalance.csv", index=False)

    oh_df = pd.read_csv("final_data/oh_concatenated.csv")
    print(len(oh_df.columns) - 38)

