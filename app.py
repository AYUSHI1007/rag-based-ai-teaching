from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from process_incoming import ask_question

app = FastAPI()

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML templates
templates = Jinja2Templates(directory="templates")


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html"
    )

# Request body model
class Question(BaseModel):
    question: str


# API endpoint
@app.post("/ask")
async def ask(data: Question):

    answer = ask_question(data.question)

    return {
        "answer": answer
    }