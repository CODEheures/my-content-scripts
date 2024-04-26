import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.decomposition import PCA
from app.collaborative import Collaborative


class ContentBased():
    """Class pour préparer les données, entrainer le model et prédire des recommandations content based
    """

    def __init__(self, artifact_path='artifacts', top_count=5) -> None:
        self.artifacts_path = artifact_path
        self.top_count = top_count
        self.content_similarities = f'{self.artifacts_path}/content_similarities.npy'


    def prepare_df(self, df_meta_data: pd.DataFrame, df_embeddings: pd.DataFrame, n_components=None, sample=None, verbose=False) -> pd.DataFrame:
        """Préparation du DataFrame d'entrainement depuis le dataframe des embeddings 
           (voir class Data pour obtenir df_meta_data et df_embeddings)

        Args:
            df_meta_data (pd.DataFrame): Dataframe des meta data
            df_embeddings (pd.DataFrame): Dataframe des embeddings
            n_components (int, optional): Nombre de composantes de la PCA. Defaults to None.
            sample (int, optional): Nombre d'embedding à garder. Defaults to None.
            verbose (bool, optional): Pour avoir des retours en terminal. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe préparé pour la fonction train
        """
        # Concat meta data with embeddings
        df_embeddings.columns = df_embeddings.columns.astype(str)
        df = pd.concat([df_meta_data, df_embeddings], axis=1)

        # Sampling
        if sample:
            df = df.sample(sample, random_state=1234)

        # Scale timestanp and word_count
        scaler = StandardScaler()
        scaled_columns = ['created_at_ts', 'words_count']
        scaled = scaler.fit_transform(df[scaled_columns])
        df[scaled_columns] = scaled

        # One hot encode categories and publisher ids
        df = pd.get_dummies(df, columns=['category_id', 'publisher_id'], dtype=int)
        
        # Apply PCA to reduce shape
        if n_components:
            pca = PCA(n_components=n_components)
            df = pd.DataFrame(pca.fit_transform(df), columns=[f'PCA_{i}' for i in range(1,len(pca.explained_variance_)+1)])
            if verbose:
                cum_sum = np.cumsum(pca.explained_variance_ratio_)
                pd.Series(cum_sum).plot()
                for i in [0.8, 0.9, 0.95]:
                    args_where = np.argwhere(np.array(cum_sum)>i)
                    if len(args_where) > 0:
                        print(f'{i*100}% de variance exprimée avec {args_where[0][0]} composantes')

        return df
    
    
    def train(self, df: pd.DataFrame, chunck_size=500, verbose=False) -> np.ndarray:
        """Entrainement du modèle content based

        Args:
            df (pd.DataFrame): Dataframe préparé par la fonction prepare_df
            chunck_size (int, optional): Découper l'entrainement par x embeddings. Defaults to 500.
            verbose (bool, optional): Pour avoir des retour en terminal. Defaults to False.

        Returns:
            np.ndarray : Matrice de similarité. Chaque ligne (article) renvoi 5 colonnes qui chacune contient l'index/score de l'article similaire
        """
        # see https://heartbeat.comet.ml/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831
        # We must chunk train because matrix of cosine similarities may cause memory overflow up to 600Gb!
        
        results = None

        # Train by chunck_size lines of df to prevent MemoryError on cosine_similarities
        for content_index in range(0,len(df),chunck_size):
           
            # Range to train
            range_start = content_index
            range_end = content_index+chunck_size
            if verbose:
                print(f'train from {range_start} to {range_end}', end='\r')
            
            current_range = df[range_start:range_end]

            # Calc cosine similarities on current range
            cosine_similarities = linear_kernel(current_range, df)
            
            # Sorting index of similiraties
            cosine_index_sort = np.argsort(-cosine_similarities, axis=1)
            
            # keep top 6 (6 to keep 5 at end of process because self similarity will be remove later)
            top_args = cosine_index_sort[:, 0:self.top_count+1]            
            
            # Remove self similiraty, and add similarity score in result
            filtered_results = np.zeros(shape=(top_args.shape[0], top_args.shape[1]-1, 2))
            for row_index in range(len(top_args)):
                filtered_result = [[v, cosine_similarities[row_index, v]]  for v in top_args[row_index] if v != row_index]
                filtered_results[row_index, 0:len(filtered_result)] = filtered_result[0:self.top_count]

            if results is None:
                results = filtered_results
            else:
                results = np.concatenate((results, filtered_results), axis=0)

        # Saving
        np.save(self.content_similarities, results)
   
        return results
    

    def recommand(self, user_id:int, verbose=True) -> list[int]:
        """Fonction de recommandation content based pour un user

        Args:
            user_id (int): Id du user à recommander
            verbose (bool, optional): Pour avoir des infos en terminal. Defaults to True.

        Returns:
            list[int]: Liste de recommandation
        """

        # Get user articles
        collaborative = Collaborative()
        user_articles = collaborative.get_user_articles(user_id=user_id)

        # Read similarities
        content_similarities = np.load(self.content_similarities)

        # keep user article only if exist on content_similarities
        filtered_user_articles = [article for article in user_articles if article < content_similarities.shape[0]]

        # DataFrame from similarities
        df_similarities = pd.DataFrame(content_similarities[filtered_user_articles, :].reshape(-1, 2), columns=['item_id', 'similarity'])
        df_similarities['item_id'] = df_similarities['item_id'].astype(int)
        
        # Filter DataFrame to remove current user article (to not recommand already readed articles)
        df_similarities = df_similarities[~df_similarities['item_id'].isin(user_articles)]

        # Sort top 5 articles by similarity score
        top_recommand = list(df_similarities.sort_values(by='similarity', ascending=False)['item_id'].unique()[0:content_similarities.shape[1]])
        
        if verbose:
            print(f"Current user articles: {', '.join(map(str, user_articles))}")
            print(f"Recommended user articles: {', '.join(map(str, top_recommand))}")

        return top_recommand
