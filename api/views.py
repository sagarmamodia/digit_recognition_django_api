import joblib
import os
from django.http import JsonResponse
from rest_framework.decorators import api_view

model = joblib.load(os.path.join(os.getcwd(), "api/predictor.joblib"))

def home(request):
    return JsonResponse({"message": "It is working!"})

@api_view(["POST"])
def predict(request):
    instance = request.data["instance"]
    prediction = model.predict([instance])
    response = {"prediction": int(prediction[0])}

    return JsonResponse(response)

