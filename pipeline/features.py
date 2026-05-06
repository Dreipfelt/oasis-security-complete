
def make_features(df):
    df = df.copy()
    df["annee"] = df["annee"].astype(int)
    df["ville_enc"] = df["ville"].astype("category").cat.codes
    df["crime_enc"] = df["crime"].astype("category").cat.codes
    return df
