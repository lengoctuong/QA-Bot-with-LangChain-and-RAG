import sys
import uvicorn
from fastapi import FastAPI
from app.api import router

app = FastAPI()
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8080)