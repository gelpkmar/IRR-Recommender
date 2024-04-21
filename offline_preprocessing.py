import pandas as pd
import os, csv
import re

def find_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def quote_urls(new_filename):
    with open(new_filename, 'r', newline='', encoding='utf-8') as input_file, \
        open("../movie_dataset/u.item_new.csv", 'w', newline='', encoding='utf-8') as output_file:

        # Initialize CSV reader and writer
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)

        # Process each row in the input CSV file
        for row in csv_reader:
            # Initialize a list to store modified row data
            modified_row = []

            # Iterate through each cell in the row
            for cell in row:
                # Find URLs in the cell
                urls = find_urls(cell)

                # If URLs are found, quote the entire cell content
                if urls:
                    cell = f'"{cell}"'

                # Append the modified cell to the modified row list
                modified_row.append(cell)

            # Write the modified row to the output CSV file
            csv_writer.writerow(modified_row)

def preprocess_text_files(directory, file, header, delimeter):
    for filename in os.listdir(directory):
        if filename == file:
            # Rename file to use ".tsv" suffix
            new_filename = filename + '.csv'
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            # Write header line with predefined column names separated by tabs
            with open(os.path.join(directory, new_filename), 'r+', errors='ignore') as file:
                content = file.read()
                file.seek(0, 0)
                file.write(header + '\n' + content)
            # Replace existing delimiters with ','
            with open(os.path.join(directory, new_filename), 'r') as file:
                content = file.read()
                # Replacing possible delimeters
                if delimeter == ',':
                    content = content.replace(',', ',')
                    content = content.replace('|', ',')
                    content = content.replace('   ', ',')
                    content = content.replace('\t', ',')
                if delimeter == ';':
                    content = content.replace('|', ';')
                    content = content.replace('   ', ';')
                    content = content.replace('\t', ';')
            with open(os.path.join(directory, new_filename), 'w') as file:
                file.write(content)

# Function calls to preprocess the text files
## Preprocess u.user
header = '\t'.join(['user_id', 'age', 'gender', 'occupation', 'zip_code'])
preprocess_text_files('../movie_dataset/', 'u.user', header, ';')

## Preprocess u.ratings
header = '\t'.join(['user_id', 'item_id', 'rating', 'timestamp'])
preprocess_text_files('../movie_dataset/', 'u.ratings', header, ';')

## Preprocess u.item
header = '\t'.join(['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
preprocess_text_files('../movie_dataset/', 'u.item', header, ';')

print("Preprocessing complete.")