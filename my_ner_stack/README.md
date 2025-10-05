# NER Stack - PubMed Central Text Extraction

This script extracts full text from PubMed Central articles using the BioC API for Named Entity Recognition (NER) processing.

## Setup

1. **Install dependencies:**
   ```bash
   pip install pandas requests python-dotenv
   ```

2. **Configure API Key:**
   - Copy `.env.example` to `.env`
   - Add your NCBI API key to the `.env` file:
     ```
     NCBI_API_KEY=your_actual_api_key_here
     ```
   - Get an API key from: https://www.ncbi.nlm.nih.gov/account/

3. **Prepare input data:**
   - Create a CSV file named `pmcids_for_api_extraction.csv`
   - The file should contain a column named `PMCID` with PubMed Central IDs

## Usage

Run the script:
```bash
python preprocessing.py
```

The script will:
- Load PMCIDs from the CSV file
- Extract full text using the BioC API
- Process the first 10 PMCIDs (configurable in code)
- Print confirmation messages for each processed article

## Features

- ✅ Environment-independent (works locally, not just in Colab)
- ✅ Secure API key management using `.env` file
- ✅ Rate limiting (0.15s pause between requests)
- ✅ Error handling for missing files and API failures
- ✅ Handles 404 responses for missing articles

## Next Steps

Add your NER pipeline code in the designated section at the end of `preprocessing.py`.
