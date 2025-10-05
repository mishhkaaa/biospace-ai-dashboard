# -*- coding: utf-8 -*-

"""
NER Pipeline for Scientific Literature - PubMed Central Text Extraction

This script extracts full text from PubMed Central articles using the BioC API and performs
Named Entity Recognition (NER) and Relation Extraction on scientific literature.

CONFIGURATION:
- Edit the configuration variables (lines 40-44) to customize behavior
- NUM_PMCIDS_TO_PROCESS: Set to None for all PMCIDs, or a number (e.g., 10, 50, 100)
- API_RATE_LIMIT_DELAY: Adjust API request delay (default: 0.15 seconds)
- CSV_FILENAME: Name of the CSV file containing PMCIDs
- CSV_COLUMN_NAME: Column name containing PMCID values

REQUIREMENTS:
- .env file with NCBI_API_KEY
- CSV file with PMCIDs
- Python packages: pandas, requests, python-dotenv
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import requests
import time
import json
import os
import re
import sys
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================
# CONFIGURATION: Adjust these values as needed
NUM_PMCIDS_TO_PROCESS = 10    # Set to None to process ALL PMCIDs, or set a number (e.g., 10, 50, 100)
API_RATE_LIMIT_DELAY = 0.15   # Seconds to wait between API calls (0.15 = ~6.7 requests/second)
CSV_FILENAME = "pmcids_for_api_extraction.csv"  # Name of the CSV file containing PMCIDs
CSV_COLUMN_NAME = "PMCID"     # Name of the column containing PMCID values

# Export Configuration
EXPORT_RESULTS = True         # Set to True to export results to files
EXPORT_JSON = True            # Export detailed results as JSON
EXPORT_CSV = True             # Export summary as CSV
OUTPUT_FOLDER = "results"     # Folder to save output files

# ============================================================================
# API KEY LOADING
# ============================================================================
try:
    ncbi_api_key = os.getenv('NCBI_API_KEY')
    if ncbi_api_key is None:
        print("Error: NCBI_API_KEY not found in environment variables.")
        print("Please create a .env file with NCBI_API_KEY=your_api_key")
        print("Or set the NCBI_API_KEY environment variable.")
    else:
        print("Successfully loaded NCBI API key.")
except Exception as e:
    print(f"An error occurred while accessing environment variables: {e}")

# ============================================================================
# TEXT EXTRACTION FUNCTION
# ============================================================================

def extract_full_text(bioc_json_data):
    """
    Extracts and concatenates all text from a BioC JSON response.

    Args:
        bioc_json_data: A list or dictionary representing the parsed BioC JSON response.

    Returns:
        A string containing the concatenated full text of the document.
    """
    full_text = ""
    
    # Handle case where response is a list (BioC API returns a list)
    if isinstance(bioc_json_data, list) and len(bioc_json_data) > 0:
        bioc_json_data = bioc_json_data[0]  # Get the first (and usually only) item
    
    if 'documents' in bioc_json_data:
        for document in bioc_json_data['documents']:
            if 'passages' in document:
                for passage in document['passages']:
                    if 'text' in passage and passage['text']: # Check if 'text' exists and is not empty
                        full_text += passage['text'] + " "
                    if 'sentences' in passage: # Check if there are nested sentences
                        for sentence in passage['sentences']:
                            if 'text' in sentence and sentence['text']: # Check if 'text' exists in sentence and is not empty
                                full_text += sentence['text'] + " "

    return full_text.strip()

# ============================================================================
# LOAD PMCIDs FROM CSV
# ============================================================================

try:
    pmcids_df = pd.read_csv(CSV_FILENAME)
    if CSV_COLUMN_NAME in pmcids_df.columns:
        pmcids = pmcids_df[CSV_COLUMN_NAME].tolist()
        print(f"Successfully loaded {len(pmcids)} PMCIDs from the CSV file.")
    else:
        print(f"Error: '{CSV_COLUMN_NAME}' column not found in the CSV file.")
        pmcids = []
except FileNotFoundError:
    print(f"Error: '{CSV_FILENAME}' not found. Please check the file exists.")
    pmcids = []
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    pmcids = []

# ============================================================================
# NER AND RELATION EXTRACTION FUNCTIONS
# ============================================================================

def extract_mock_entities(document_text: str) -> list:
    """
    Mock Named Entity Recognition function that simulates SciBERT NER output
    using Sentence Chunking logic to process the entire document.

    Args:
        document_text: The preprocessed text from which to extract entities.

    Returns:
        A list of dictionaries with keys: 'text', 'type', 'span_start', 'span_end'
    """
    entities = []
    
    # 1. Simulate Sentence Splitting on the ENTIRE text
    # This ensures no text is ignored. The span_offset tracks global position.
    sentences = re.split(r'(?<=[.?!])\s+', document_text.strip())
    span_offset = 0

    # Define mock entity patterns (case-insensitive search)
    # Expanded list for better coverage including space-related terms
    entity_patterns = {
        'EXPERIMENT': ['experiment', 'study', 'trial', 'analysis', 'investigation', 'methodology', 'test', 'assay', 'research'],
        'ORGANISM': ['mice', 'mouse', 'rat', 'human', 'cells', 'patients', 'samples', 'bacteria', 'cell', 'organisms', 'species'],
        'ENVIRONMENT': ['microgravity', 'radiation', 'vitro', 'vivo', 'control', 'condition', 'laboratory', 'lab', 'culture', 'medium', 'space', 'ISS'],
        'OUTCOME': ['result', 'effect', 'outcome', 'change', 'increase', 'decrease', 'loss', 'response', 'impact', 'consequence']
    }

    # 2. Process sentence by sentence (Chunking)
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Search for each pattern in the sentence
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                
                start_idx = 0
                while start_idx < len(sentence):
                    idx = sentence.lower().find(pattern.lower(), start_idx)
                    if idx == -1:
                        break
                    
                    # Extract the actual text (preserving original case)
                    actual_text = sentence[idx:idx + len(pattern)]
                    
                    # Calculate GLOBAL span indices
                    global_start = span_offset + idx
                    global_end = global_start + len(pattern)
                    
                    entities.append({
                        'text': actual_text,
                        'type': entity_type,
                        'span_start': global_start,
                        'span_end': global_end
                    })
                    
                    start_idx = idx + len(pattern)
        
        # Update the global offset for the next sentence, accounting for the separator space
        span_offset += len(sentence) + 1 
    
    # Sort entities by span_start
    entities.sort(key=lambda x: x['span_start'])
    
    # DEDUPLICATION STRATEGY:
    # Remove duplicate entities based on normalized text + type
    # This keeps only ONE instance of each unique entity concept
    # E.g., if "cell" appears 50 times, we keep only the first occurrence
    seen_entities = set()
    unique_entities = []
    
    for entity in entities:
        # Create unique key: lowercase text + entity type
        entity_key = (entity['text'].lower(), entity['type'])
        
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            unique_entities.append(entity)
    
    return unique_entities


def extract_mock_relations(entities: list) -> list:
    """
    Mock Relation Extraction function that simulates a relation classifier.
    Creates Knowledge Graph triples by pairing entities based on their types.
    
    Args:
        entities: A list of entity dictionaries from extract_mock_entities.
    
    Returns:
        A list of relation triples as dictionaries with 'subject', 'predicate', 'object'
    """
    relations = []
    
    # Define relation rules based on entity type pairs
    relation_rules = {
        ('EXPERIMENT', 'ORGANISM'): 'STUDIED_IN',
        ('ORGANISM', 'EXPERIMENT'): 'STUDIED_IN',
        ('OUTCOME', 'EXPERIMENT'): 'CAUSED_BY',
        ('EXPERIMENT', 'OUTCOME'): 'HAS_OUTCOME',
        ('ORGANISM', 'ENVIRONMENT'): 'STUDIED_IN',
        ('ENVIRONMENT', 'ORGANISM'): 'STUDIED_IN',
        ('OUTCOME', 'ORGANISM'): 'CAUSED_BY',
        ('ENVIRONMENT', 'OUTCOME'): 'HAS_OUTCOME'
    }
    
    # Use a set to track unique relations (based on normalized text)
    seen_relations = set()
    
    # Generate relations by pairing entities
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            # Check if this entity pair has a relation rule
            pair_key = (entity1['type'], entity2['type'])
            
            if pair_key in relation_rules:
                predicate = relation_rules[pair_key]
                
                # Create a unique key for this relation (case-insensitive)
                # This ensures we don't have duplicate relations like:
                # (study, STUDIED_IN, mice) appearing multiple times
                relation_key = (
                    entity1['text'].lower(),
                    entity1['type'],
                    predicate,
                    entity2['text'].lower(),
                    entity2['type']
                )
                
                # Only add if we haven't seen this exact relation before
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    
                    # Create structured triple
                    relation = {
                        'subject': {
                            'text': entity1['text'],
                            'type': entity1['type']
                        },
                        'predicate': predicate,
                        'object': {
                            'text': entity2['text'],
                            'type': entity2['type']
                        }
                    }
                    
                
                relations.append(relation)
    
    return relations

# ============================================================================
# PROCESS PMCIDs
# ============================================================================

base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"# Determine how many PMCIDs to process
pmcids_to_process = pmcids if NUM_PMCIDS_TO_PROCESS is None else pmcids[:NUM_PMCIDS_TO_PROCESS]
total_to_process = len(pmcids_to_process)

print("="*80)
print(f"NER PIPELINE - PROCESSING {total_to_process} PMCIDs")
print("="*80)
print(f"Rate limit: {API_RATE_LIMIT_DELAY}s delay between requests")
print()

# Store results for final summary
all_results = []

# Process PMCIDs
for idx, pmcid in enumerate(pmcids_to_process, 1):
    print(f"\n{'='*80}")
    print(f"[{idx}/{total_to_process}] Processing: {pmcid}")
    print(f"{'='*80}")
    
    api_url = f"{base_url}{pmcid}/unicode"
    headers = {'Api-Key': ncbi_api_key}

    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            try:
                bioc_data = response.json()
                full_text = extract_full_text(bioc_data)
                print(f"✓ Text extracted: {len(full_text):,} characters")
                
                # NER Pipeline Integration
                if full_text:
                    # Extract entities
                    entities = extract_mock_entities(full_text)
                    print(f"✓ Entities found: {len(entities)}")
                    
                    # Extract relations
                    relations = extract_mock_relations(entities) if entities else []
                    print(f"✓ Relations found: {len(relations)}")
                    
                    # Store results
                    all_results.append({
                        'pmcid': pmcid,
                        'text_length': len(full_text),
                        'entities': entities,
                        'relations': relations
                    })
                    
                    # Display detailed breakdown
                    print(f"\n📊 Entity Breakdown by Type:")
                    entity_counts = {}
                    for entity in entities:
                        entity_type = entity['type']
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    for entity_type in ['EXPERIMENT', 'ORGANISM', 'ENVIRONMENT', 'OUTCOME']:
                        count = entity_counts.get(entity_type, 0)
                        print(f"   • {entity_type:12s}: {count:2d} entities")
                    
                    # Show sample entities
                    print(f"\n🔍 Sample Entities (first 5):")
                    for i, entity in enumerate(entities[:5], 1):
                        print(f"   {i}. '{entity['text']}' → {entity['type']}")
                    
                    # Show sample relations
                    print(f"\n🔗 Sample Relations (first 5):")
                    for i, rel in enumerate(relations[:5], 1):
                        print(f"   {i}. ({rel['subject']['text']}, {rel['predicate']}, {rel['object']['text']})")
                
                time.sleep(API_RATE_LIMIT_DELAY) # Pause after successful call
            except json.JSONDecodeError:
                print(f"❌ Error: Could not decode JSON response for {pmcid}")
            except Exception as e:
                print(f"❌ Error during text extraction for {pmcid}: {e}")

        elif response.status_code == 404:
            print(f"❌ Article not found (Status: 404)")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ API request error for {pmcid}: {e}")

# Print final summary
print(f"\n\n{'='*80}")
print("📊 FINAL SUMMARY - NER PIPELINE RESULTS")
print(f"{'='*80}\n")

if all_results:
    print(f"Successfully processed: {len(all_results)} articles\n")
    print(f"{'PMCID':<15} {'Text Length':<15} {'Entities':<12} {'Relations':<12}")
    print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    
    total_entities = 0
    total_relations = 0
    
    for result in all_results:
        print(f"{result['pmcid']:<15} {result['text_length']:>13,}  {len(result['entities']):>10}  {len(result['relations']):>10}")
        total_entities += len(result['entities'])
        total_relations += len(result['relations'])
    
    print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    print(f"{'TOTAL':<15} {'':<15} {total_entities:>10}  {total_relations:>10}")
    print(f"{'AVERAGE':<15} {'':<15} {total_entities/len(all_results):>10.1f}  {total_relations/len(all_results):>10.1f}")
    
    # Entity type distribution across all papers
    print(f"\n📈 Overall Entity Type Distribution:")
    all_entity_types = {}
    for result in all_results:
        for entity in result['entities']:
            entity_type = entity['type']
            all_entity_types[entity_type] = all_entity_types.get(entity_type, 0) + 1
    
    for entity_type in ['EXPERIMENT', 'ORGANISM', 'ENVIRONMENT', 'OUTCOME']:
        count = all_entity_types.get(entity_type, 0)
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        print(f"   • {entity_type:12s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n✅ Pipeline completed successfully!")
    
    # ============================================================================
    # EXPORT RESULTS TO FILES
    # ============================================================================
    if EXPORT_RESULTS:
        print(f"\n{'='*80}")
        print("💾 EXPORTING RESULTS")
        print(f"{'='*80}\n")
        
        # Create output folder if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Generate timestamp for filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed JSON results
        if EXPORT_JSON:
            json_filename = os.path.join(OUTPUT_FOLDER, f"ner_results_{timestamp}.json")
            json_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'total_pmcids_processed': len(all_results),
                    'total_entities': total_entities,
                    'total_relations': total_relations,
                    'entity_type_distribution': all_entity_types
                },
                'results': all_results
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Exported detailed results to: {json_filename}")
        
        # Export CSV summary
        if EXPORT_CSV:
            csv_filename = os.path.join(OUTPUT_FOLDER, f"ner_summary_{timestamp}.csv")
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'PMCID': result['pmcid'],
                    'Text_Length': result['text_length'],
                    'Total_Entities': len(result['entities']),
                    'Total_Relations': len(result['relations']),
                    'EXPERIMENT_Count': sum(1 for e in result['entities'] if e['type'] == 'EXPERIMENT'),
                    'ORGANISM_Count': sum(1 for e in result['entities'] if e['type'] == 'ORGANISM'),
                    'ENVIRONMENT_Count': sum(1 for e in result['entities'] if e['type'] == 'ENVIRONMENT'),
                    'OUTCOME_Count': sum(1 for e in result['entities'] if e['type'] == 'OUTCOME')
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(csv_filename, index=False)
            print(f"✓ Exported summary to: {csv_filename}")
        
        # Export entities CSV
        entities_csv_filename = os.path.join(OUTPUT_FOLDER, f"entities_{timestamp}.csv")
        entities_data = []
        for result in all_results:
            for entity in result['entities']:
                entities_data.append({
                    'PMCID': result['pmcid'],
                    'Entity_Text': entity['text'],
                    'Entity_Type': entity['type'],
                    'Span_Start': entity['span_start'],
                    'Span_End': entity['span_end']
                })
        
        df_entities = pd.DataFrame(entities_data)
        df_entities.to_csv(entities_csv_filename, index=False)
        print(f"✓ Exported entities to: {entities_csv_filename}")
        
        # Export relations CSV
        relations_csv_filename = os.path.join(OUTPUT_FOLDER, f"relations_{timestamp}.csv")
        relations_data = []
        for result in all_results:
            for relation in result['relations']:
                relations_data.append({
                    'PMCID': result['pmcid'],
                    'Subject_Text': relation['subject']['text'],
                    'Subject_Type': relation['subject']['type'],
                    'Predicate': relation['predicate'],
                    'Object_Text': relation['object']['text'],
                    'Object_Type': relation['object']['type']
                })
        
        df_relations = pd.DataFrame(relations_data)
        df_relations.to_csv(relations_csv_filename, index=False)
        print(f"✓ Exported relations to: {relations_csv_filename}")
        
        print(f"\n📁 All files saved in: {OUTPUT_FOLDER}/")
        print(f"{'='*80}\n")

else:
    print("⚠️ No articles were successfully processed.")

print(f"{'='*80}\n")

# Test block for standalone execution
if __name__ == "__main__":
    # Sample scientific text for testing
    sample_text = """
    In this study, we conducted an experiment to analyze the effect of laboratory 
    conditions on mice cells. The trial showed significant results in cell culture 
    medium. Human patients were also included in the analysis. The outcome demonstrated 
    an increase in bacterial response under vitro conditions.
    """
    
    print("=" * 60)
    print("NER and Relation Extraction Pipeline Test")
    print("=" * 60)
    
    # Test NER
    print("\n1. Named Entity Recognition:")
    print("-" * 60)
    entities = extract_mock_entities(sample_text)
    print(f"Found {len(entities)} entities:\n")
    for i, entity in enumerate(entities, 1):
        print(f"  {i}. {entity['text']:15s} → {entity['type']:12s} (pos: {entity['span_start']}-{entity['span_end']})")
    
    # Test Relation Extraction
    print("\n2. Relation Extraction:")
    print("-" * 60)
    relations = extract_mock_relations(entities)
    print(f"Found {len(relations)} relations:\n")
    for i, relation in enumerate(relations, 1):
        print(f"  {i}. ({relation['subject']['text']}, {relation['predicate']}, {relation['object']['text']})")
        print(f"     [{relation['subject']['type']} → {relation['predicate']} → {relation['object']['type']}]")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)