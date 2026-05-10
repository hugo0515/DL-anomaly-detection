import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def load_and_preprocess_unsw(train_path, test_path):
    # 1. Load UNSW-NB15 dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Drop unnecessary columns
    drop_cols = ["id", "attack_cat"]

    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # 3. Create validation set from training data
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )

    # 4. Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"].values

    X_val = val_df.drop("label", axis=1)
    y_val = val_df["label"].values

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"].values

    # 5. Identify categorical and numerical columns
    categorical_cols = ["proto", "service", "state"]

    numerical_cols = [
        col for col in X_train.columns
        if col not in categorical_cols
    ]

    # 6. One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore")

    X_train_cat = encoder.fit_transform(X_train[categorical_cols]).toarray()
    X_val_cat = encoder.transform(X_val[categorical_cols]).toarray()
    X_test_cat = encoder.transform(X_test[categorical_cols]).toarray()

    # 7. Normalize numerical features
    scaler = MinMaxScaler()

    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_val_num = scaler.transform(X_val[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    # 8. Combine numerical + encoded categorical features
    X_train_final = np.hstack((X_train_num, X_train_cat))
    X_val_final = np.hstack((X_val_num, X_val_cat))
    X_test_final = np.hstack((X_test_num, X_test_cat))

    print("UNSW-NB15 preprocessing completed.")
    print(f"X_train shape: {X_train_final.shape}")
    print(f"X_val shape:   {X_val_final.shape}")
    print(f"X_test shape:  {X_test_final.shape}")

    return (
        X_train_final,
        y_train,
        X_test_final,
        y_test,
        X_val_final,
        y_val
    )
