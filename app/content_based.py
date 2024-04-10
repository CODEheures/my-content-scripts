import pandas as pd
import pickle
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class ContentBased():

    def __init__(self, artifact_path='artifacts') -> None:
        self.artifacts_path = artifact_path
        self.df_path = f'{self.artifacts_path}/df_content_based.pickle'
        self.model_path = f'{self.artifacts_path}/model_content_based.pickle'


    def prepare_df(self, df_meta_data: pd.DataFrame, df_embeddings: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df_meta_data, df_embeddings], axis=1)

        # Scale timestanp and word_count
        scaler = StandardScaler()
        scaled_columns = ['created_at_ts', 'words_count']
        scaled = scaler.fit_transform(df[scaled_columns])
        df[scaled_columns] = scaled

        # One hot encode categories and publisher ids
        df = pd.get_dummies(df, columns=['category_id', 'publisher_id'])

        return df
    
    
    def train(self, df: pd.DataFrame, verbose=False) -> SVD:

        trainset, testset = train_test_split(df, test_size=0.3)
        # https://heartbeat.comet.ml/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831
        
        if verbose:
            predictions = model.test(testset)
            accuracy.rmse(predictions)

        df.to_pickle(self.df_path)
        pickle.dump(model, open(self.model_path, 'wb'))
        return model
    
    
    def predict(self, row, model):
        return model.predict(row.uid, row.iid)


    def get_users(self) -> list[int]:
        df = pd.read_pickle(self.df_path)
        return list(df.sample(10)['user_id'])


    def user_exist(self, user_id) -> bool:
        df = pd.read_pickle(self.df_path)
        return len(df.loc[df['user_id'] == user_id]['user_id'].unique()) == 1


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
