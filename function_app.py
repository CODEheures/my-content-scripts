import azure.functions as func
import logging
from app.collaborative import Collaborative
from app.content_based import ContentBased
from app.data import Data
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="recommand", methods=["GET"])
def recommand(req: func.HttpRequest) -> func.HttpResponse:

    user_id = req.params.get('userId')
    
    if user_id:
        try:
            user_id = int(user_id)
        except:
            return func.HttpResponse(f"userId doit être un nombre entier")
        collaborative = Collaborative()
        content_based = ContentBased()
        if (collaborative.user_exist(user_id=user_id)):
            logging.info(f'Calcul recommandations pour le user {user_id}')
            recommandations_collaborative = collaborative.recommand(user_id=user_id, verbose=False)
            recommandations_content_based = content_based.recommand(user_id=user_id, verbose=False)
            return func.HttpResponse(
                        json.dumps({'collaborative': [str(x) for x in recommandations_collaborative], 'content_based': [str(x) for x in recommandations_content_based]}),
                        mimetype="application/json",
                 )
        else:
            return func.HttpResponse(
             "Cet utilisateur n'existe pas",
             status_code=404
        )
    else:
        return func.HttpResponse(
             "Merci d'ajouter un userId à votre requête ?userId=xxxx",
             status_code=400
        )
    
@app.route(route="train/collaborative", methods=["GET"])
def train_collaborative(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info(f'Entrainement modèle collaboratif demandé')
    data = Data()
    collaborative = Collaborative()
    sample_users_size = 20000
    
    logging.info(f'Lecture des fichiers clicks en cours...')
    df_clicks = data.read_clicks()
    sample_df = collaborative.prepare_df(df_clicks=df_clicks, max_clicks=5, n_users=sample_users_size)
    
    logging.info(f'Entrainement en cours...')
    model = collaborative.train(sample_df, verbose=False)
    
    return func.HttpResponse(
             "Entrainement du model collaborative terminé",
             status_code=200
        )

@app.route(route="train/content-based", methods=["GET"])
def train_content_based(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info(f'Entrainement modèle content based demandé')
    data = Data()
    content_based = ContentBased()
    sample = 20000
    
    logging.info(f'Lecture des fichiers meta data et embeddings en cours...')
    df_meta_data = data.read_articles_meta_data()
    df_embeddings = data.read_embeddings()

    logging.info(f'Preparation du dataframe...')
    df = content_based.prepare_df(df_meta_data=df_meta_data, df_embeddings=df_embeddings, n_components=43, sample=sample)
    
    logging.info(f'Entrainement en cours...')
    results = content_based.train(df=df, chunck_size=500)
    
    return func.HttpResponse(
             "Entrainement du model content based terminé",
             status_code=200
        )