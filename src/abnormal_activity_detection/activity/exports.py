"""Writing activity-classifier outputs."""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from abnormal_activity_detection.data import ACTIVITY_INDEX

INDEX_ACTIVITY = {v: k for k, v in ACTIVITY_INDEX.items()}


def export_features_to_csv_with_vec(vec: DictVectorizer,
                                    X_dicts, y, groups, meta,
                                    out_path: str):
    """Write the vectorised features, with metadata and labels, to a CSV."""
    X = vec.transform(X_dicts)
    feature_names = vec.get_feature_names_out()
    dfX = pd.DataFrame(X.toarray(), columns=feature_names)

    meta_df = pd.DataFrame(meta).reset_index(drop=True)
    df = pd.concat([meta_df, dfX], axis=1)

    df.insert(0, "y_index", y.astype(int))
    df.insert(1, "y_label", [INDEX_ACTIVITY[int(i)] for i in y])

    df.to_csv(out_path, index=False)
    print(f"✓ Exported {len(df)} rows -> {out_path}  (cols={df.shape[1]})")
