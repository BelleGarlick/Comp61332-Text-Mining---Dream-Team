import numpy as np
def load_glove(path):
    """
    Load pretrained GloVe word embedding from specified path.

    Given a path this function open, reads and creates a dictionary that converts each word into 
    a Numpy array of the respective word embedding. Also extracts the size of these word embeddings
    to be use in filling in missing embeddings.

    Args:
        path: A path in the format of the string to the embedding file.

    Returns:
        A tuple of the word embedding dictionary and an integer size of each word embedding.
    """
    words, vectors, word2idx, idx = [], [], {}, 0
    with open (path) as f:
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

def embed(sentences): 
    """
    Embeds the provided list of sentences into the GloVe embedding space.

    Given a list of sentences, represented by an array of words, convert each word into their respective 
    Numpy array in the GloVe embedding space. Replacing words that are missing from the specified pretrained 
    word embedding with a random Numpy array.

    Args:
        sentences: A list where each item is sentence represented by a list of words.

    Returns:
        An embeddings list of lists for each sentence with a numpy arrray representing each word.
    """
    glove, emb_dim = load_glove("../data/glove.small.txt")
    embedings = []

    for sentence in sentences:
        for word in sentence:
            embeding = []
            try:
                embeding.append(glove[word])
            except KeyError:
                embeding.append(np.random.normal(scale=0.6, size=(emb_dim, )))
        # TODO Hyperparamater what to do with special tags specified in the tokenizer 
        # Either a random array, array of zeros or use the word in the tag i.e. for "#date#" use "date"
        embedings.append(embeding)

    return embedings

    
    


# 00 out or add #data as date, hyperparamarter