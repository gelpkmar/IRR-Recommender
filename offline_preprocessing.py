import pandas as pd
import os, csv
import re

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

def create_test_set(source_file, destination_file, delim):
    # Creation of TestSet to evaluate recommendation engine performance
    ratings_df = pd.read_csv(source_file, delimiter=delim)

    selected_columns_df = ratings_df[['user_id', 'item_id']].head(2000)

    # Write the DataFrame to a new CSV file with the specified delimiter
    selected_columns_df.to_csv(destination_file, sep=delim, index=False)

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

## Preprocess items.csv
header = ['item_id','movie_title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western','Summary','Cast','Director','Rating','Runtime','No. of ratings','YT-Trailer ID']
### Load the CSV file
df = pd.read_csv('../movie_dataset/items.csv')
### Change the header row to the custom line
df.columns = header
### Save back to .csv
df.to_csv('../movie_dataset/items.csv', index=False)


## Create Test Set
create_test_set('../movie_dataset/u.ratings.csv', '../movie_dataset/selected_ratings.csv', ";")




print("Preprocessing complete.")