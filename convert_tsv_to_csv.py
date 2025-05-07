import pandas as pd

# File paths for TSV and CSV files
completeDataset_tsv_file = '.\\completeDataset.tsv'
testSet_tsv_file = '.\\testSet.tsv'
trainingSet_tsv_file = '.\\trainingSet.tsv'

completeDataset_1_csv_file = '.\\rdf_triples_completeDataset.csv'
testSet_1_csv_file = '.\\rdf_triples_testSet.csv'
trainingSet_1_csv_file = '.\\rdf_triples_trainingSet.csv'

# Define the predicate (assumed for all rows)
predicate = "http://swrc.ontoware.org/ontology#affiliation"

# Function to convert TSV to RDF triples in CSV format
def convert_tsv_to_rdf_triples(tsv_file, csv_file):
    # Load the TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Create a new DataFrame for RDF triples
    rdf_triples = pd.DataFrame({
        "subject": df["person"],  # Map 'person' column to 'subject'
        "predicate": predicate,   # Use the same predicate for all rows
        "object": df["label_affiliation"]  # Map 'label_affiliation' column to 'object'
    })
    
    # Save the RDF triples to a new CSV file
    rdf_triples.to_csv(csv_file, index=False)
    print(f"Converted {tsv_file} to RDF triples in {csv_file}")

# Convert each TSV file to RDF triples
convert_tsv_to_rdf_triples(completeDataset_tsv_file, completeDataset_1_csv_file)
convert_tsv_to_rdf_triples(testSet_tsv_file, testSet_1_csv_file)
convert_tsv_to_rdf_triples(trainingSet_tsv_file, trainingSet_1_csv_file)