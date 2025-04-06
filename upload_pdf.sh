#!/bin/bash

# Define variables
PDF_PATH="data/full_contract_pdf/Part_I/License_Agreements/DataCallTechnologies_20060918_SB-2A_EX-10.9_944510_EX-10.9_Content License Agreement.pdf"
API_URL="http://localhost:8000/upload"

# Check if the PDF file exists
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF file not found at $PDF_PATH"
    exit 1
fi

# Upload the PDF file
echo "Uploading PDF file..."
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "file=@$PDF_PATH" \
    $API_URL

echo -e "\nDone!" 