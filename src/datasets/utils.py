import pandas as pd
from definitions import *
import re

def read_raw_files(parts=[0,1,2,3,4,5,6], columns = ["note_id",
           "subject_id",
           "hadm_id_",
           "note_type",
           "note_seq",
           "charttime",
           "storetime",
           "text"]):
    
    prefix = "00000000000"
    dfs = []
    for i in range(len(parts)):
        filename = RAW_DATA_PATH + prefix + str(parts[i])
        df = pd.read_csv(filename, header=None)
        df.columns = columns
        dfs.append(df)

    final_df = pd.concat(dfs)
    return final_df

# def write_interim_files():
#     df = read_raw_files()
#     write_path = INTERIM_DATA_PATH + "df.csv"
#     df.to_csv(write_path)
#     return 0 

# def read_interim_files():
#     read_path = INTERIM_DATA_PATH + "df.csv"
#     df = pd.read_csv(read_path)
#     return df

# def write_processed_files():
#     read_fp = INTERIM_DATA_PATH + "df.csv"
#     df = pd.read_csv(read_fp)
#     notes = df.text
#     concatenated_notes = notes.str.cat(sep=('\n'))
#     processed_notes = concatenated_notes.replace("\n", " ") ## TODO BASED ON EDA



def get_unique_chars(text):
    unique_chars = ''.join(set(text))
    chars = [x.encode('ascii', 'ignore').decode("utf-8") for x in unique_chars]
    chars = set(sorted(chars))
    return chars

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace("\n", " ")
    return text

def clean_text_chunked(text, chunk_size=10000):
    cleaned_text = ""
    start = 0
    n_chunks = len(text)/chunk_size + 1
    while start < len(text):
        chunk = text[start:start+chunk_size]
        cleaned_chunk = clean_text(chunk)
        cleaned_text += cleaned_chunk
        start += chunk_size
    return cleaned_text

def write_interim_files():
    prefix = "00000000000"
    for i in range(0, 7):
        print("Starting to process chunk {}".format(str(i)))
        df = read_raw_files(parts=[i])
        notes = df.text
        concatenated_notes = notes.str.cat(sep='\n')
        cleaned_notes = clean_text_chunked(concatenated_notes)
        write_path = INTERIM_DATA_PATH + "cleaned_notes_" + prefix + str(i) + ".txt"
        with open(write_path, "w+") as f:
            f.write(cleaned_notes)
        print("Finished writing chunk {}".format(str(i)))