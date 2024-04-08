import os
import pandas as pd

class Data():

    def __init__(self, path='data') -> None:
        self.path = path

    def read_clicks(self) -> pd.DataFrame:
        df = None
        clicks_path = f'{self.path}/clicks'
        for f in os.listdir(clicks_path):
            current_df = pd.read_csv(f"{clicks_path}/{f}", sep=",", header=0)
            df = pd.concat([df, current_df], axis=0, ignore_index=True)
        return df
    
    def read_articles_meta_data(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self.path}/articles_metadata.csv", sep=",", header=0, index_col=0)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def read_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame(pd.read_pickle(f'{self.path}/articles_embeddings.pickle'))
