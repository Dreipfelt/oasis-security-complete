
import pandas as pd

def load_clean(path):
    df = pd.read_excel(path)
    df = df.iloc[2:]
    df = df.rename(columns={
        df.columns[0]:"annee",
        df.columns[1]:"code",
        df.columns[2]:"crime"
    })

    df = df.melt(
        id_vars=["annee","code","crime"],
        var_name="ville",
        value_name="nb"
    )

    df = df.dropna(subset=["nb"])
    df["nb"] = pd.to_numeric(df["nb"], errors="coerce")
    df = df.dropna()
    return df
