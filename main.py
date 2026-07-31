from fastapi import FastAPI
app=FastAPI()

@app.get('/')
def greet():
    return {'message':'vyapardock backend is live'}