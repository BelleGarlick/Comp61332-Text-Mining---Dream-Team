

def load(path: str):
    """
    Load questions and questions types from a path.

    This function will, given a path to the dataset file, read in, line-by-line, a text document extracting questions
    and questions types. The document is structured such that the QType is the first token of the questions. This
    function reads each line, splits up the tokens and extracts the first token from the line placing it in the types
    list.

    # TODO Pre-process the word embeddings.

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
            questions.append(" ".join(tokens[1:]))

    # TODO pre-process data

    return questions, types
