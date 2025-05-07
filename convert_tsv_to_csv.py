import rdflib
import random
import pandas as pd

# completeDataset_tsv_file = '.\\completeDataset.tsv'
testSet_tsv_file= '.\\testSet.tsv'
trainingSet_tsv_file= '.\\trainingSet.tsv'

# completeDataset_csv_file='.\completeDataset.csv'
testSet_csv_file='.\\testSet.csv'    
trainingSet_csv_file='.\\trainingSet.csv'


def convert_tsv_to_csv(tsv_file, csv_file):
    df = pd.read_csv(tsv_file)  
    df.to_csv(csv_file, index=False)      
    print(f"Converted {tsv_file} to {csv_file}")

# Convert each TSV file to CSV
# convert_tsv_to_csv(completeDataset_tsv_file, completeDataset_csv_file)
# convert_tsv_to_csv(testSet_tsv_file, testSet_csv_file)
# convert_tsv_to_csv(trainingSet_tsv_file, trainingSet_csv_file)
