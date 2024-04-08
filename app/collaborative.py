import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pickle
from pandarallel import pandarallel


class Collaborative():

    def __init__(self, artifact_path='artifacts') -> None:
        self.artifacts_path = artifact_path
        self.df_path = f'{self.artifacts_path}/df_collaborative.pickle'
        self.model_path = f'{self.artifacts_path}/model_collaborative.pickle'


    def aggregate_clicks(self, df_clicks: pd.DataFrame, max_clicks=None, n_users=None) -> pd.DataFrame:
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
            df = df[df['user_id'].isin(random_users_list)]

        return df
    
    
    def train(self, df: pd.DataFrame, rating_scale=(1,5), verbose=False) -> SVD:

        reader = Reader(rating_scale=rating_scale)

        data = Dataset.load_from_df(df[["user_id", "article_id", "nb_clicks"]], reader)
        trainset, testset = train_test_split(data, test_size=0.3)

        algo = SVD()
        algo.fit(trainset=trainset)

        if verbose:
            predictions = algo.test(testset)
            accuracy.rmse(predictions)

        df.to_pickle(self.df_path)
        pickle.dump(algo, open(self.model_path, 'wb'))
        return algo
    
    
    def predict(self, row, model):
        return model.predict(row.uid, row.iid)


    def get_users(self) -> list[int]:
        df = pd.read_pickle(self.df_path)
        return list(df.sample(10)['user_id'])

    
    def recommand(self, user_id:int, verbose=True) -> list[int]:
        pandarallel.initialize(progress_bar=verbose)

        df = pd.read_pickle(self.df_path)
        model = pickle.load(open(self.model_path, 'rb'))

        user_articles = list(df.loc[df['user_id'] == user_id]['article_id'].unique())
        not_viewed_article_ids = list(df.loc[~df['article_id'].isin(user_articles)]['article_id'].unique())
        results = pd.DataFrame([[user_id, article_id, 0] for article_id in not_viewed_article_ids], columns=['uid', 'iid', 'est'])

        results['est'] = results.parallel_apply(self.predict, args=(model,), axis=1)

        results['iid'] = results['iid'].astype('int64')

        results.sort_values(by='est', ascending=False, inplace=True, ignore_index=True)

        top_recommand = list(results.loc[0:5, 'iid'])
        
        if verbose:
            print(f"Current user articles: {', '.join(map(str, user_articles))}")
            print(f"Recommended user articles: {', '.join(map(str, top_recommand))}")

        return top_recommand
