import requests
import time
import sys

def get_document_status(document_id):
    url = f"http://localhost:8000/api/v1/documents/{document_id}/status"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting document status: {e}")
        return None

def monitor_document_processing(document_id):
    print(f"Monitoring document ID: {document_id}")
    while True:
        status_data = get_document_status(document_id)
        if status_data:
            current_status = status_data.get('status')
            print(f"Current status: {current_status}")
            
            if current_status in ["CHUNKING_COMPLETED", "FAILED", "DUPLICATE"]:
                print("Processing completed or failed.")
                print("Final status details:")
                print(status_data)
                break
        else:
            print("Could not retrieve status, retrying...")
        
        time.sleep(5) # Wait for 5 seconds before polling again

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_status.py <document_id>")
        sys.exit(1)
    
    doc_id_to_monitor = sys.argv[1]
    monitor_document_processing(doc_id_to_monitor)
