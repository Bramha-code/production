import re
import uuid
from pathlib import Path

def patch_ingestion_service(file_path: Path):
    with open(file_path, "r") as f:
        content = f.read()

    # Pattern to find the process_upload method and its content
    process_upload_pattern = re.compile(
        r"(async def process_upload(self,[\s\S]*?)\s*-> DocumentUploadResponse:[\s\S]*?)(\s*return DocumentUploadResponse([\s\S]*?)\s*)\n\s*async def",
        re.DOTALL
    )
    
    match = process_upload_pattern.search(content)

    if match:
        before_return = match.group(1)
        return_block = match.group(2)
        after_process_upload = content[match.end():]

        # Ensure doc_id generation uses uuid.uuid4()
        if "doc_id = str(uuid.uuid4())" not in before_return:
            before_return = re.sub(
                r"(doc_id = content_hash[:12])",
                r"        doc_id = str(uuid.uuid4()) # Generate a UUID for the document",
                before_return
            )

        # Ensure database.create_document(metadata) is present
        if "await self.database.create_document(metadata)" not in before_return:
            before_return = re.sub(
                r"(metadata = DocumentMetadataDB([\s\S]*?s3_raw_path=s3_path,\s*))",
                r"\1\n\n        await self.database.create_document(metadata)",
                before_return
            )
        
        # Ensure message_broker.publish_document_uploaded is present
        if "await self.message_broker.publish_document_uploaded(" not in before_return:
            before_return = re.sub(
                r"(await self.database.create_document(metadata))",
                r"\1\n\n        # Publish event\n        await self.message_broker.publish_document_uploaded(\n            doc_id=doc_id,\n            s3_path=s3_path,\n            file_size=file_size,\n            content_hash=content_hash,\n        )",
                before_return
            )

        new_process_upload_content = before_return + return_block

        # Reconstruct the file content
        content = process_upload_pattern.sub(new_process_upload_content + "\n\s*async def", content, 1)

    # Apply changes to handle_processing_failure, get_document_status, retry_document
    # These functions already appear to be correctly implemented based on previous reads,
    # so we'll just ensure the imports are there.

    # Ensure uuid import
    if "import uuid" not in content:
        content = re.sub(r"(import asyncio)", r"\1\nimport uuid", content)

    # Ensure Optional is imported
    if "from typing import Optional" not in content:
        content = re.sub(r"(import asyncio)", r"from typing import Optional\n\1", content)
    
    # Ensure DocumentUploadResponse has file_size
    if "file_size: int" not in content:
        content = re.sub(r"(content_hash: str)", r"\1\n    file_size: int", content)


    with open(file_path, "w") as f:
        f.write(content)

    print(f"Patched {file_path}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    ingestion_service_path = script_dir.parent / "services" / "ingestion_service.py"
    patch_ingestion_service(ingestion_service_path)
