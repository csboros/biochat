def convert_to_csv(input_file, output_file):
    import csv
    
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
input_file = '/taxon.txt'
output_file = './taxon.csv'
convert_to_csv(input_file, output_file)