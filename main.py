import sys
import uvicorn
from fastapi import FastAPI
from app.api import router

fastapp = FastAPI()
fastapp.include_router(router)

if __name__ == '__main__':
    uvicorn.run(fastapp, host='0.0.0.0', port=7890)