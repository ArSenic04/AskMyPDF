import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF for PDF processing
from transformers import pipeline

# Flask configuration
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the QA model (e.g., DistilBERT or similar)
qa_pipeline = pipeline('question-answering')
extracted_text = ""  # This will store the text extracted from the PDF

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(filepath):
    global extracted_text
    extracted_text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            extracted_text += page.get_text()
    return extracted_text

# Route for uploading PDF files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            extracted_text_content = extract_text_from_pdf(filepath)  # Extract and store the text
            
            # Print the filename and extracted text to the terminal
            print(f"File '{file.filename}' submitted successfully.")
            print("Extracted Text:")
            print(extracted_text_content)
            
            return jsonify({'message': f"File '{file.filename}' submitted successfully and text extracted."}), 200
        except Exception as e:
            return jsonify({'message': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Invalid file format. Only PDF files are allowed.'}), 400

# Route to answer questions based on extracted text
@app.route('/ask', methods=['POST'])
def ask_question():
    global extracted_text
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'message': 'Question not provided'}), 400
    
    if not extracted_text:
        return jsonify({'message': 'No text extracted from PDF. Please upload a PDF first.'}), 400

    response = qa_pipeline({'question': question, 'context': extracted_text})
    answer = response['answer']
    return jsonify({'answer': answer}), 200

if __name__ == '__main__':
    app.run(port=5000)
