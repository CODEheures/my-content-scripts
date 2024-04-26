import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pickle
from pandarallel import pandarallel


class Collaborative():
    """Class pour préparer les données, entrainer le model et prédire des recommandations collaboratives
    """
    def __init__(self, artifact_path='artifacts') -> None:
        self.artifacts_path = artifact_path
        self.df_path = f'{self.artifacts_path}/df_collaborative.pickle'
        self.model_path = f'{self.artifacts_path}/model_collaborative.pickle'


    def prepare_df(self, df_clicks: pd.DataFrame, max_clicks=None, n_users=None) -> pd.DataFrame:
        """Préparation du DataFrame d'entrainement depuis le dataframe des clicks 
           (voir class Data pour obtenir df_click)

        Args:
            df_clicks (pd.DataFrame): Dataframe des clicks obtenu par la class Data
            max_clicks (int, optional): Nombre de click à partir duquel le rating est maxi. Defaults to None.
            n_users (int, optional): Nombre de users maxi pour limiter l'entrainement. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe prêt pour la fonction d'entrainement
        """
        df = df_clicks.loc[:, ['user_id', 'click_article_id', 'session_id']]
        df = df.groupby(['user_id', 'click_article_id']).count()
        df.reset_index(inplace=True)
        df.rename(columns={'click_article_id': 'article_id', 'session_id': 'nb_clicks'}, inplace=True)

        if max_clicks is not None:
            df.loc[df['nb_clicks'] >=max_clicks, 'nb_clicks'] = max_clicks

        if n_users is not None:
            np.random.seed(1234)
            full_users_list = df['user_id'].unique()
            random_users_list = np.random.choice(full_users_list, size=min(n_users, len(full_users_list)))
            np.random.seed(None)
            df = df[df['user_id'].isin(random_users_list)]

        return df
    
    
    def train(self, df: pd.DataFrame, rating_scale=(1,5), verbose=False) -> SVD:
        """Entrainement du model

        Args:
            df (pd.DataFrame): Dataframe préparé par la fonction prepare_df
            rating_scale (tuple, optional): Note mini-maxi. Defaults to (1,5).
            verbose (bool, optional): Pour avoir des retour terminal. Defaults to False.

        Returns:
            SVD: Model collaboratif
        """
        reader = Reader(rating_scale=rating_scale)

        data = Dataset.load_from_df(df[["user_id", "article_id", "nb_clicks"]], reader)
        trainset, testset = train_test_split(data, test_size=0.3)

        model = SVD()
        model.fit(trainset=trainset)

        if verbose:
            predictions = model.test(testset)
            accuracy.rmse(predictions)

        df.to_pickle(self.df_path)
        pickle.dump(model, open(self.model_path, 'wb'))
        return model
    
    
    def predict(self, row, model):
        """Prédiction sur le model obtenu depuis la fonction train
           Invoquée depuis la fonction recommand
        """
        return model.predict(row.uid, row.iid)


    def get_users(self, nb_users=10) -> list[int]:
        """Obtenir une liste de users entrainés

        Args:
            nb_users (int, optional): Nombre de users. Defaults to 10.

        Returns:
            list[int]: Liste des users
        """
        df = pd.read_pickle(self.df_path)
        return list(df.sample(nb_users)['user_id'])
    
    
    def get_user_articles(self, user_id: int, df: pd.DataFrame = None) -> list[int]:
        """Obtenir la liste des articles d'un user

        Args:
            user_id (int): User id
            df (pd.DataFrame, optional): Dataframe ayant servi à l'entrainement du model. Defaults to None.

        Returns:
            list[int]: Liste des articles
        """
        if df is None:
            df = pd.read_pickle(self.df_path)

        return list(df.loc[df['user_id'] == user_id]['article_id'].unique())


    def user_exist(self, user_id) -> bool:
        """Defini si un user existe dans le dataframe d'entrainement

        Args:
            user_id (int): User Id à tester

        Returns:
            bool: True si le user a été entrainé
        """
        df = pd.read_pickle(self.df_path)
        return len(df.loc[df['user_id'] == user_id]['user_id'].unique()) == 1


    def recommand(self, user_id:int, verbose=True) -> list[int]:
        """Obtenir une liste de recommadations collaborative pour un user

        Args:
            user_id (int): Id du user à recommander
            verbose (bool, optional): Obtenir des retour en terminal. Defaults to True.

        Returns:
            list[int]: Liste recommandée
        """
        pandarallel.initialize(progress_bar=verbose)

        df = pd.read_pickle(self.df_path)
        model = pickle.load(open(self.model_path, 'rb'))

        user_articles = self.get_user_articles(user_id=user_id, df=df)
        not_viewed_article_ids = list(df.loc[~df['article_id'].isin(user_articles)]['article_id'].unique())
        results = pd.DataFrame([[user_id, article_id, 0] for article_id in not_viewed_article_ids], columns=['uid', 'iid', 'est'])

        results['est'] = results.parallel_apply(self.predict, args=(model,), axis=1)

        results['iid'] = results['iid'].astype('int64')

        results.sort_values(by='est', ascending=False, inplace=True, ignore_index=True)

        top_recommand = list(results['iid'].unique()[0:5])
        
        if verbose:
            print(f"Current user articles: {', '.join(map(str, user_articles))}")
            print(f"Recommended user articles: {', '.join(map(str, top_recommand))}")

        return top_recommand
