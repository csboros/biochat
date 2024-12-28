"""Script to convert tab-separated taxonomic data to CSV format."""

import csv

def convert_to_csv(input_file, output_file):
    """Convert tab-separated taxonomic data file to CSV format.
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output CSV file
    """
    # Define the headers based on the file structure
    headers = [
        'id', 'scientific_name', 'kingdom', 'phylum', 'class', 'order_name', 
        'family_name', 'genus_name', 'species_name', 'authority', 'rank', 'name_status', 
        'accepted_id', 'id2', 'reference', 'url'
    ]
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            # Write the headers
            writer.writerow(headers)
            # Process each line
            for line in infile:
                # Split the line by tabs
                fields = line.strip().split('\t')
                writer.writerow(fields)

# Usage
INPUT_FILE = '/taxon.txt'
OUTPUT_FILE = './taxon.csv'
convert_to_csv(INPUT_FILE, OUTPUT_FILE)
