from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple API Demo")

class MessageRequest(BaseModel):
    text: str

class MessageResponse(BaseModel):
    original: str
    reversed: str
    length: int
    uppercase: str

@app.get("/")
def root():
    return {"message": "FastAPI is running! Send POST to /process"}

@app.post("/process", response_model=MessageResponse)
def process_message(request: MessageRequest):
    """
    Simple endpoint that processes a text message.
    This demonstrates the basic FastAPI <-> Streamlit connection.
    """
    text = request.text
    
    return MessageResponse(
        original=text,
        reversed=text[::-1],
        length=len(text),
        uppercase=text.upper()
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}