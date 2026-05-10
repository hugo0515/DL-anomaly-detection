# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from config import TRAIN_PATH, TEST_PATH


def load_and_preprocess(train_path=TRAIN_PATH, test_path=TEST_PATH):
    # Column names
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    train_df = pd.read_csv(train_path, names=columns)
    test_df = pd.read_csv(test_path, names=columns)

    # Drop difficulty
    train_df = train_df.drop("difficulty", axis=1)
    test_df = test_df.drop("difficulty", axis=1)

    # Split features & labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Categorical & numeric
    categorical_cols = ["protocol_type", "service", "flag"]
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    # One-hot encoding
    encoder = OneHotEncoder()
    X_train_cat = encoder.fit_transform(X_train[categorical_cols]).toarray()
    X_test_cat = encoder.transform(X_test[categorical_cols]).toarray()

    # Normalization
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    # Combine features
    X_train_final = np.hstack((X_train_num, X_train_cat))
    X_test_final = np.hstack((X_test_num, X_test_cat))

    # preprocess.py (add after X_train_final and X_test_final)

    X_train_final = np.array(X_train_final)
    X_test_final = np.array(X_test_final)

    return X_train_final, y_train.values, X_test_final, y_test.values
