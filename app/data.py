import os
import pandas as pd

class Data():
    """Class permettant de lire les Data depuis les fichiers clicks/metadata/embeddings
    """

    def __init__(self, path='data') -> None:
        self.path = path

    def read_clicks(self) -> pd.DataFrame:
        """Obtenir le dataframe df_clicks depuis les fichiers clicks

        Returns:
            pd.DataFrame: Dataframe des clicks
        """
        df = None
        clicks_path = f'{self.path}/clicks'
        for f in os.listdir(clicks_path):
            current_df = pd.read_csv(f"{clicks_path}/{f}", sep=",", header=0)
            df = pd.concat([df, current_df], axis=0, ignore_index=True)
        return df
    
    def read_articles_meta_data(self) -> pd.DataFrame:
        """Obtenir le dataframe df_meta_data depuis le fichier des meta data

        Returns:
            pd.DataFrame: Dataframe des meta data
        """
        df = pd.read_csv(f"{self.path}/articles_metadata.csv", sep=",", header=0, index_col=0)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def read_embeddings(self) -> pd.DataFrame:
        """Obtenir le dataframe df_embeddings depuis le fichier des embeddings

        Returns:
            pd.DataFrame: Dataframe des embeddings
        """
        return pd.DataFrame(pd.read_pickle(f'{self.path}/articles_embeddings.pickle'))
