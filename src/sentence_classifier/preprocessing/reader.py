def load(path: str):
    """
    Load questions and questions types from a path.

    This function will, given a path to the dataset file, read in, line-by-line, a text document extracting questions
    and questions types. The document is structured such that the QType is the first token of the questions. This
    function reads each line, splits up the tokens and extracts the first token from the line placing it in the types
    list.

    Args:
        path: A path in the format of the string to the document file.

    Returns:
        A tuple of the questions and targets stored as ([q1, q2, ... qn], [qtype1, qtype2, ... qtypen])
    """

    # Read and split data
    questions, types = [], []
    with open(path) as file:
        for line in file.readlines():
            tokens = line.split()
            types.append(tokens[0])
            questions.append(tokens[1:])

    return questions, types

def load_pretrained(path):
    """
    Load pretrained GloVe word embedding from specified path.

    Given a path this function open, reads and creates a dictionary that converts each word into 
    a Tensor array of the respective word embedding. Also extracts the size of these word embeddings
    to be use in filling in missing embeddings.

    Args:
        path: A path in the format of the string to the embedding file.

    Returns:
        A tuple of the word embedding dictionary and an integer size of each word embedding.
    """
    words, vectors, word2idx, idx = [], [], {}, 0
    with open(path) as f:
        for l in f:
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx+=1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    
    glove = {w: vectors[word2idx[w]] for w in words}

    return glove, vectors[0].shape[0]

