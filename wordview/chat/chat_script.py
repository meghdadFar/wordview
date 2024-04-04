import json

import pandas as pd

from wordview.text_analysis import TextStatsPlots

if __name__ == "__main__":
    imdb_df = pd.read_csv("data/IMDB_Dataset_sample_5k.csv")
    with open("wordview/chat/secrets/openai_api_key.json", "r") as f:
        credentials = json.load(f)

    tsp = TextStatsPlots(df=imdb_df, text_column="review")
    tsp.chat(api_key=credentials.get("openai_api_key"))
