from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()

@app.post("/process/")
async def process_file_and_string(
    file: UploadFile = File(...),
    string_input: str = Form(...)
):
    # Read the file
    file_content = await file.read()
    # Do something with the file and string
    result = {
        "file_name": file.filename,
        "file_size": len(file_content),
        "string_input": string_input,
        "message": f"File '{file.filename}' processed with input '{string_input}'"
    }
    return result