from preprocessing import load, parse_tokens


# Rules used during tokenisation.
tokenisation_rules = {
    "TOKENISE_QUOTES": True,
    "TOKENISE_NUMBERS": True,
    "TOKENISE_MONTH": True,
    "TOKENISE_PERCENTAGES": True,
    "REMOVE_QUESTION_MARKS": True,
    "TOKENISE_STOPWORDS": False,
    "TOKENISE_URLS": True,
    "TOKENISE_MONEY": True,
    "TOKENISE_YEAR": True,
    "TOKENISE_COMMA_SEPERATED_NUMBERS": True,
    "TOKENISE_LOWER_CASE": True
}


if __name__ == "__main__":
    # Load the dataset
    questions, classifications = load("../data/train.txt")

    # Map questions to tokenised questions
    tokenised_questions = list(map(lambda x: parse_tokens(x, tokenisation_rules), questions))

    # Display the pre and post tokenised questions
    for i, question in enumerate(questions):
        print(f"{question}\n{tokenised_questions[i]}\n{classifications[i]}\n")
