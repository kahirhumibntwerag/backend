from fastapi import FastAPI

# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startups")


@app.get("/api/search")
def search_startup(q: str):
    return {"result": neural_searcher.search(text=q)}


