import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from argparse import ArgumentParser
import ipdb

LOG = True  # disable if don't want console output


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    if LOG:
        print("Read CSV from", csv_path)
        print(df.info(verbose=True))
    return df


def get_columns_from_file(file):
    with open(file) as f:
        columns = f.readlines()
    columns = [x.strip() for x in columns]
    return columns


def split_columns_xy(df, y_columns):
    y_df = df[y_columns]
    x_df = df.drop(y_columns, axis=1)
    return x_df, y_df


def select_columns(df, column_list):
    feat_cols = [c for c in list(df.columns) if c[: c.rfind("_")] in column_list]
    df = df[feat_cols]
    if LOG:
        print("Columns selected from the input CSV:", df.columns)
    return df


def reduce_dimensions(input_df, n_components):
    print("Initiating PCA with {} components".format(n_components))
    pca = PCA(n_components)
    pca.fit(input_df)
    if LOG:
        print(
            "PCA explained_variance:\n",
            pca.explained_variance_,
            "\n",
            "PCA explained_variance_ratio_:\n",
            pca.explained_variance_ratio_,
            "PCA retained variance:\n",
            np.sum(pca.explained_variance_ratio_),
        )
    input_df_compressed = pca.transform(input_df)
    return input_df_compressed


def save_file(data, destination, is_input):
    destination = destination + "_input" if is_input else destination + "_label"
    if isinstance(data, pd.DataFrame):
        destination = destination + ".csv"
        print(f"Saving label DataFrame of size {len(data)} ")
        data.to_csv(destination)
    else:
        print(f"Saving pca represenation of size {data.shape} ")
        destination = destination + ".npy"
        with open(destination, "wb") as f:
            np.save(f, data)
    if LOG:
        print("Saved to", destination)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="final_data/oh_concatenated.csv",
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--label_columns_file",
        type=str,
        default="final_data/sample_label_columns.txt",
        help="Text file with list of label columns. These columns will be excluded in PCA.",
    )
    parser.add_argument(
        "--enable_column_selection",
        action="store_true",
        help="If true, a list of columns will be selected from a text file",
    )
    parser.add_argument(
        "--columns_text_file",
        type=str,
        default="final_data/sample_selected_columns.txt",
        help="Text file with a list of columns to use for PCA.",
    )
    parser.add_argument(
        "--use_pca", action="store_true", default=True, help="Use PCA to reduce the dimensionality"
    )
    parser.add_argument(
        "--pca_ncomponents",
        type=int,
        default=64,
        help="Number of components wanted in the compressed data",
    )
    parser.add_argument(
        "--destination_file", type=str, default="pca_comp_data", help="Path to the output file",
    )
    args = parser.parse_args()

    df = read_csv(args.input_csv)

    # split columns into input and target columns
    label_columns = get_columns_from_file(args.label_columns_file)
    x_columns, y_columns = split_columns_xy(df, label_columns)

    # select columns, optionally
    if args.enable_column_selection:
        selected_columns = get_columns_from_file(args.columns_text_file)
        x_columns = select_columns(x_columns, selected_columns)

    # reduce dimensions, optionally
    if args.use_pca:
        x_columns = reduce_dimensions(x_columns, args.pca_ncomponents)

    # save the compressed data to disk
    save_file(x_columns, args.destination_file, is_input=True)
    save_file(y_columns, args.destination_file, is_input=False)
