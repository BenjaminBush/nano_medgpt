import pandas as pd
from definitions import *
import re
from src.features.features import *
import torch
import pickle

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

def get_unique_chars(text):
    unique_chars = ''.join(set(text))
    chars = [x.encode('ascii', 'ignore').decode("utf-8") for x in unique_chars]
    chars = set(sorted(chars))
    return chars

def clean_text(text):
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode("utf-8")
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

def read_interim_files(parts=[0,1,2,3,4,5,6]):
    notes = ""
    prefix = "00000000000"
    for i in range(0, 7):
        fname = INTERIM_DATA_PATH + "cleaned_notes_" + prefix + str(i) + ".txt"
        with open(fname, "r") as f:
            notes += f.readlines()[0]
    return notes

def train_test_split(ratio=0.8, notes=None):
    train_path = PROCESSED_DATA_PATH + "train.pt"
    test_path = PROCESSED_DATA_PATH + "test.pt"

    # If the data exists, then just read and return. 
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        train = torch.load(train_path)
        test = torch.load(test_path)
    
    # Else, we need to generate and then write then the data to disk. 
    else:
        if notes is None:
            notes = read_interim_files()
       
        n = len(notes)

        # Encoding the notes will run into a memory error, so we need to chunk again
        encodings = []
        chunk_size = 1000000
        start = 0
        i = 0
        n_chunks = n/chunk_size + 1
        while start < n:
            chunk = notes[start:start+chunk_size]
            encoded_chunk = encode(chunk)
            fpath = INTERIM_DATA_PATH + "encodings/chunk_" + str(start) + ".pkl"
            with open(fpath, 'wb') as f:
                pickle.dump(encoded_chunk, f)
            
            start += chunk_size

            i+=1
            if i % 100 == 0:
                print("Written {} out of {} chunks".format(i, n_chunks))

        # Need to come back to fix this. Encodings should be all of the pkl files
        # with open(file, 'rb') as f:
        #   enc = pickle.load(f)
        # encodings += enc
        encodings = encoded_chunk

        # Creat the dataset
        dataset = torch.tensor(encodings, dtype=torch.long)

        # Split into train/test sets
        split = int(ratio*n)
        train = dataset[:split]
        test = dataset[split:]

        torch.save(train, train_path)
        torch.save(test, test_path)
    
    return train, test

