from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from rag_pipeline import initialize_pipeline, generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the actual origin of your frontend for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
retriever = None
llm = None


class Query(BaseModel):
    question: str


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever, llm
    try:
        # Save the uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Initialize the pipeline with the uploaded PDF
        retriever, llm = initialize_pipeline(file_location)

        # Delete the temporary file after processing
        os.remove(file_location)

        return {"message": "PDF processed successfully. You can now ask questions."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/")
async def ask_question(query: Query):
    global retriever, llm
    if retriever is None or llm is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded and processed.")

    try:
        answer = generate_answer(retriever, llm, query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
