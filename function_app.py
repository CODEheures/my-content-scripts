import azure.functions as func
import logging
from app.collaborative import Collaborative
from app.data import Data

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
        if (collaborative.user_exist(user_id=user_id)):
            logging.info(f'Calcul recommandations pour le user {user_id}')
            recommandations = collaborative.recommand(user_id=user_id, verbose=False)
            return func.HttpResponse(f"Les autres utilisateurs ont aussi aimé: {' '.join(map(str, recommandations))}.")
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
    collaborative.train(sample_df, verbose=False)
    return func.HttpResponse(
             "Entrainement du model collaborative terminé",
             status_code=200
        )