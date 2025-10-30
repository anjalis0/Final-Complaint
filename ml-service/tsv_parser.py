import pandas as pd



df = pd.read_csv(r"train_sentiment.tsv", sep='\t')
df.to_csv(r"train_sentiment.txt", index=False, sep="|")
