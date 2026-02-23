import pandas as pd

df = pd.read_csv("data/intent_data.csv")
print("Original class counts:")
print(df["label"].value_counts())

# Simple startegy: duplicate crisis rows N times
N = 6   # tune: 6 copies -> increases crisis weight
crisis_rows = df[df["label"] == "crisis"]
rest = df[df["label"] != "crisis"]
df_new = pd.concat([rest, pd.concat([crisis_rows]*N, ignore_index=True)], ignore_index=True)
df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
df_new.to_csv("data/intent_data_oversampled.csv", index=False)
print("New class counts:")
print(df_new["label"].value_counts())
print("Saved to data/intent_data_oversampled.csv")
