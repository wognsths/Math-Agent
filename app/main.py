from fastapi import FastAPI
from api.routes.latex import router as latex_router
import uvicorn

app = FastAPI(title="Math Agent API")

# Register routers
app.include_router(latex_router)

@app.get("/")
async def root():
    return {"message": "Math Agent API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 