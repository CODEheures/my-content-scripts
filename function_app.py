import azure.functions as func
import logging
from app.collaborative import Collaborative

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="recommand", methods=["GET"])
def recommand(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('userId')
    
    if user_id:
        try:
            user_id = int(user_id)
        except:
            return func.HttpResponse(f"userId doit être un nombre entier")
        collaborative = Collaborative()
        if (collaborative.user_exist(user_id=user_id)):
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