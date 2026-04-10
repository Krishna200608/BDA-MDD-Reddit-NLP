import pandas as pd
import os

base_dir = r"d:\Lab\BDA\Project"

for filename in ['reddit_raw.csv', 'reddit_mdd_cleaned.csv']:
    path = os.path.join(base_dir, 'data', 'processed' if 'cleaned' in filename else 'raw', filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'subreddit' in df.columns:
            df.loc[df['subreddit'] == 'depression', 'label'] = 'Moderate MDD'
            df.loc[df['subreddit'] == 'SuicideWatch', 'label'] = 'Severe Ideation'
            df.to_csv(path, index=False)
            print(f"Migrated labels for {filename}")
    else:
        print(f"File not found: {path}")
