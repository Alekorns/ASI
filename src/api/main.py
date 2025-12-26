from fastapi import FastAPI
from src.api.routes import predict, train, models

app = FastAPI(title="Iris API")

app.include_router(train.router)
app.include_router(predict.router)
app.include_router(models.router)
