
char_vocab = list("0123456789abcdefghijklmnopqrstuvwxyz ")
stoi = {}
itos = {}
i = 0
for c in char_vocab:
    stoi[c] = i
    itos[i] = c
    i+=1
    
# Encode: string -> int
encode = lambda s: [stoi[c] for c in s]

# Decode: int -> string
decode = lambda i: ''.join([itos[j] for j in i])
