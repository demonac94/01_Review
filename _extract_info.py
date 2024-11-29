import os
import re
import pandas as pd

#================================================================================================
# Create a CSV file with abstracts extracted from .bib files  
#================================================================================================

def extract_abstracts_bib(bib_filename, input_dir, output_dir):
    
    # Regular expressions to match the fields
    patterns = {
        'author': re.compile(r'author\s*=\s*{(.*?)}', re.DOTALL),
        'title': re.compile(r'title\s*=\s*{(.*?)}', re.DOTALL),
        'year': re.compile(r'year\s*=\s*{(.*?)}', re.DOTALL),
        'journal': re.compile(r'journal\s*=\s*{(.*?)}', re.DOTALL),
        'url': re.compile(r'url\s*=\s*{(.*?)}', re.DOTALL),
        'doi': re.compile(r'doi\s*=\s*{(.*?)}', re.DOTALL),
        'abstract': re.compile(r'abstract\s*=\s*{(.*?)}', re.DOTALL),
        'author_keywords': re.compile(r'author_keywords\s*=\s*{(.*?)}', re.DOTALL),
        'keywords': re.compile(r'keywords\s*=\s*{(.*?)}', re.DOTALL),
        'type': re.compile(r'type\s*=\s*{(.*?)}', re.DOTALL),
        'publication_stage': re.compile(r'publication_stage\s*=\s*{(.*?)}', re.DOTALL),
        'source': re.compile(r'source\s*=\s*{(.*?)}', re.DOTALL),
        'note': re.compile(r'note\s*=\s*{(.*?)}', re.DOTALL),
    }

    # Lists to store the extracted data
    data = {key: [] for key in patterns}

    # Function to eliminate text after a certain point
    def eliminate_after_point(text):
        return text.split(' ©')[0]

    # Function to extract a single field from an entry
    def extract_field(entry, pattern):
        entry = entry.replace('\u2013'  , '-')
        entry = entry.replace('\u2014'  , '-')
        entry = entry.replace('\u2018'  , "'")
        entry = entry.replace('−'       , "-")
        entry = entry.replace(' '       , ' ')
        entry = entry.replace('∼'       , "-")
        entry = entry.replace('∗'       , "*")
        entry = entry.replace('⋅'       ,'')
        entry = entry.replace('  '       , ' ')
        entry = entry.replace('  '       , ' ')
        match = pattern.search(entry)
        return eliminate_after_point(match.group(1)) if match else "NoData"

    # Open one bib file
    bib_path = os.path.join(input_dir, bib_filename+'.bib')
    with open(bib_path, 'r', encoding='utf-8') as bib_file:
        bib_content = bib_file.read()

        # Split the content into individual entries
        entries = bib_content.split('@ARTICLE')
        for entry in entries:
            if entry.strip():
                for key, pattern in patterns.items():
                    data[key].append(extract_field(entry, pattern))

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data)

    # Filter out rows with 'NoData' in the 'title' column
    df = df[df['title'] != 'NoData']

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a CSV file
    excel_path = os.path.join(output_dir, f'{bib_filename}_export.xlsx')

    df.to_excel(excel_path, index=False)
    info_result = 'The bib file ' + bib_filename + ' was exported successfully with ' + str(len(df)) + ' entries.'
    print(info_result)

    return excel_path