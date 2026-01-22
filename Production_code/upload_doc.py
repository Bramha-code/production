import requests
import os

def upload_document(file_path):
    url = "http://localhost:8000/api/v1/documents/upload"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(url, files=files)
        
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("Document uploaded successfully!")
        print(response.json())
        return response.json().get('document_id')
    except requests.exceptions.RequestException as e:
        print(f"Error uploading document: {e}")
        if 'response' in locals() and response is not None:
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text}")
        return None

if __name__ == "__main__":
    doc_directory = "standard_document"
    for filename in os.listdir(doc_directory):
        if filename.endswith(".pdf"):
            document_path = os.path.join(doc_directory, filename)
            print(f"Uploading {document_path}...")
            doc_id = upload_document(document_path)
            if doc_id:
                print(f"Uploaded document ID: {doc_id}")
