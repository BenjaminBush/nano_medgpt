import pandas as pd
from definitions import *
import re
from src.features.features import *
import torch
import pickle

torch.manual_seed(456123)

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

# Returns an array. Each element of the array corresponds to the notes for that part
def read_interim_files(parts=[0,1,2,3,4,5,6]):
    notes = []
    prefix = "00000000000"
    for i in range(0, 7):
        notes.append("")
        fname = INTERIM_DATA_PATH + "cleaned_notes_" + prefix + str(i) + ".txt"
        with open(fname, "r") as f:
            notes[i] += f.readlines()[0]
    return notes

def make_encodings(notes=None):
    if notes is None:
        notes = read_interim_files()
    

    # Encoding the notes will run into a memory error, so we need to chunk again
    # encodings = []
    # chunk_size = 1000000
    # for i in range(len(notes)):
    #     n = len(notes[i])
    #     start = 0
    #     while start < n:
    #         chunk = notes[i][start:start+chunk_size]
    #         encoded_chunk = encode(chunk)
    #         fpath = INTERIM_DATA_PATH + "encodings/part/" + str(i) + "/chunk_" + str(start) + ".pkl"
    #         with open(fpath, 'wb') as f:
    #             pickle.dump(encoded_chunk, f)
            
    #         start += chunk_size


    # Concatenate the chunks into whole parts
    for i in range(len(notes)):
        encodings = []
        directory = INTERIM_DATA_PATH + "encodings/part/" + str(i) + "/"
        for filename in os.listdir(directory):
            fpath = os.path.join(directory, filename)
            with open(fpath, "rb") as f:
                enc = pickle.load(f)
            encodings += enc
        out_file = PROCESSED_DATA_PATH + "part_" + str(i) + "_encoded.pt"
        dataset = torch.tensor(encodings, dtype=torch.long)
        torch.save(dataset, out_file)
