# 120kW_ETL_pipeline
120kW Extract, Tranform and Loading Pipeline using Marker-PDF

### Using Conda with Python Version = 3.10 and *activate the conda on the Workstation PC [conda activate myenv]
### Install Marker - pip install marker-pdf (also install all dependencies in python 3.10)

### Steps To Run The Complete Pipeline Follow this and for Understanding the Directory view file 'src/directory.txt':

1. Run The Marker CLI command mentioned above to get the first JSON Output.
2. Run The 'src/collect_json.py' code to collect and store all the standards documents json output in one place.
3. Run The 'src/json_to_schema.py' to convert into a structured json schema.
4. Use the structured json schema from dir 'data/output/output_schema' for chunking.
5. Use 'src/schema_to_chunks.py' to convert the JSON Schema into JSON Chunks for KG and Database.
6. Use 'src/pipeline.py' to run the entire pipeline with one command from extraction to chunking process.

set CUDA_VISIBLE_DEVICES="" # since there is not point of using GPU for marker

### CLI command for testing Marker:
marker data/input_pdfs --output_dir data/output/marker_json --workers 1 --output_format json
