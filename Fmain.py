import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import process_pdf, create_vectorstore, generate_response

app = FastAPI()

# In-memory storage for vector store
vectorstore = None

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    try:
        # Save uploaded file
        file_path = os.path.join("uploaded_files", file.filename)
        os.makedirs("uploaded_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process PDF and create vector store
        global vectorstore
        pages, splits = process_pdf(file_path)
        vectorstore = create_vectorstore(splits)
        return JSONResponse(content={"message": "PDF processed successfully!", "pages": len(pages), "chunks": len(splits)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    try:
        global vectorstore
        if not vectorstore:
            return JSONResponse(content={"error": "No PDF has been uploaded and processed yet."}, status_code=400)
        
        # Generate response
        answer = generate_response(vectorstore, question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
