from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.compose import ColumnTransformer


def load_and_combine_files(file_paths):
    """
    Load CSV files and combine them into a single DataFrame.
    Also removes ID column.
    Downcasts All Columns.
    """
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, encoding='cp1252')
        df.columns = df.columns.str.strip()
        df.drop(columns='Flow ID', inplace=True)
        df.drop(columns="Timestamp", inplace=True)

        df['Fwd PSH Flags'] = df['Fwd PSH Flags'].astype('category')
        df['Bwd PSH Flags'] = df['Bwd PSH Flags'].astype('category')
        df['Fwd URG Flags'] = df['Fwd URG Flags'].astype('category')
        df['Bwd URG Flags'] = df['Bwd URG Flags'].astype('category')
        df['FIN Flag Count'] = df['FIN Flag Count'].astype('category')
        df['SYN Flag Count'] = df['SYN Flag Count'].astype('category')
        df['RST Flag Count'] = df['RST Flag Count'].astype('category')
        df['PSH Flag Count'] = df['PSH Flag Count'].astype('category')
        df['ACK Flag Count'] = df['ACK Flag Count'].astype('category')
        df['URG Flag Count'] = df['URG Flag Count'].astype('category')
        df['CWE Flag Count'] = df['CWE Flag Count'].astype('category')
        df['ECE Flag Count'] = df['ECE Flag Count'].astype('category')
        df['Fwd Avg Bytes/Bulk'] = df['Fwd Avg Bytes/Bulk'].astype('category')
        df['Fwd Avg Packets/Bulk'] = df['Fwd Avg Packets/Bulk'].astype('category')
        df['Fwd Avg Bulk Rate'] = df['Fwd Avg Bulk Rate'].astype('category')
        df['Bwd Avg Bytes/Bulk'] = df['Bwd Avg Bytes/Bulk'].astype('category')
        df['Bwd Avg Packets/Bulk'] = df['Bwd Avg Packets/Bulk'].astype('category')
        df['Bwd Avg Bulk Rate'] = df['Bwd Avg Bulk Rate'].astype('category')
        df['Protocol'] = df['Protocol'].astype('category')
        df[['Source Port', 'Destination Port']] = df[['Source Port', 'Destination Port']].astype(int)

        df[['Source Port', 'Destination Port']] = df[['Source Port', 'Destination Port']].apply(pd.to_numeric, downcast="integer")

        df['Label'] = df['Label'].astype('category')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        dfs.append(df)
        print(f"Completed:{file_path}")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def downsample_benign(df: pd.DataFrame, label_col='Label'):
    """
    Downsamples the benign class to balance the dataset.
    Combines rare attacks (labels with <20000 samples) into 'RARE_ATTACK'.
    Drops labels with <100 samples.
    """
    counts = df[label_col].value_counts()
    # Drop labels with <100 samples
    valid_labels = counts[counts >= 100].index
    df = df[df[label_col].isin(valid_labels)]

    # Combine labels with <20000 samples into 'RARE_ATTACK'
    counts = df[label_col].value_counts()  # recalculate after dropping
    rare_labels = counts[(counts < 20000) & (counts.index != 'BENIGN')].index
    df[label_col] = df[label_col].apply(lambda x: 'RARE_ATTACK' if x in rare_labels else x)

    # Recalculate after relabeling
    benign_data = df[df[label_col] == 'BENIGN']
    attack_data = df[df[label_col] != 'BENIGN']

    # Only downsample if there are enough benign samples
    if len(benign_data) > len(attack_data) and len(attack_data) > 0:
        benign_downsampled = benign_data.sample(len(attack_data), random_state=42)
        result = pd.concat([benign_downsampled, attack_data], ignore_index=True)
    else:
        result = df  # fallback: do not downsample

    return result


def normalize_features(df, feature_cols):
    """
    Normalize the features to have mean=0 and std=1.
    """
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def process_data(file_paths, feature_cols):
    """
    Combine, downsample, and normalize the data.
    """
    df = load_and_combine_files(file_paths)
    df = downsample_benign(df)
    df = normalize_features(df, feature_cols)
    df.dropna(inplace=True)
    return df


def get_split_data(file_paths, feature_cols):
    df = process_data(file_paths, feature_cols)
    categorical_cols = ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                        'Bwd URG Flags', 'Protocol']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    X = df.drop(columns='Label')
    y = df['Label']

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)

    # Get new column names after one-hot encoding
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    passthrough_cols = [col for col in X.columns if col not in categorical_cols]
    all_feature_names = list(ohe_feature_names) + passthrough_cols

    # Create DataFrame with preserved column names
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)  # type: ignore

    return (X_transformed_df, y)
