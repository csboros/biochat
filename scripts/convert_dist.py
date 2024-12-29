"""Script to convert species distribution data from TSV to CSV format."""

import csv

def convert_to_csv(input_file, output_file):
    """Convert distribution data from tab-separated file to CSV format.
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output CSV file
    """
    # Open the input file and create the output CSV file
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        # Create CSV writer
        writer = csv.writer(outfile)
        # Write header row
        writer.writerow(['ID', 'Citation', 'Conservation_Status', 'Scope', 'Presence'])
        # Process each line
        for line in infile:
            # Split the line by tabs
            parts = line.strip().split('\t')
            # Skip empty lines
            if not parts or len(parts) < 5:
                continue
            # Write the data to CSV
            writer.writerow([
                parts[0],  # ID
                parts[1],  # Citation
                parts[2],  # Conservation Status
                parts[3],  # Scope
                parts[4]   # Presence
            ])

# Run the conversion
INPUT_FILE = './distribution.txt'
OUTPUT_FILE = './distribution.csv'
convert_to_csv(INPUT_FILE, OUTPUT_FILE)
