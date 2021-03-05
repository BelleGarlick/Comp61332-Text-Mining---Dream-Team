
"""
We haven't decided whether to alter the tokens for abbreviations yet so this class is a stub.
"""
ABBREVIATION_TOKENS = {
    "'t": {
        "replacement": "not",
        "pres": [
            ("can", "can"),
            ("didn", "did"),
            ("shouldn", "should"),
            ("doesn", "does"),
            ("wasn", "was"),
            ("isn", "is"),
            ("won", "will"),
            ("didn", "did"),
            ("shan", "shall"),
            ("haven", "have"),
            ("don", "do")
        ]
    },
    "'ve": {
        "replacement": "have",
        "pres": [
            ("you", "you")
        ]
    },
    "'d": {
        "replacement": "have",
        "pres": []
    },
    "'em": {
        "replacement": "have",
        "pres": []
    },
    "'etat": {
        "replacement": "have",
        "pres": []
    },
    "'hara": {
        "replacement": "have",
        "pres": []
    },
    "'l": {
        "replacement": "have",
        "pres": []
    },
    "'ll": {
        "replacement": "will",
        "pres": [
            ("we", "we"),
            ("he", "he"),
            ("she", "she")
        ]
    },
    "'m": {
        "replacement": "have",
        "pres": []
    },
    "'n": {
        "replacement": "have",
        "pres": []
    },
    "'neal": {
        "replacement": "have",
        "pres": []
    },
    "'re": {
        "replacement": "are",
        "pres": [
            ("they", "they"),
            ("you", "you")
        ]
    },
    # "'s": {
    #     "replacement": "have",
    #     "pres": []
    # }
}

def replace_abbreviations(tokens):
    for i in range(len(tokens)):
        if tokens[i] in ABBREVIATION_TOKENS:
            for token, replacement in ABBREVIATION_TOKENS[tokens[i]]["pres"]:
                if tokens[i - 1] == token:
                    tokens[i - 1] = replacement
                    tokens[i] = ABBREVIATION_TOKENS[tokens[i]]["replacement"]

            if tokens[i] in ABBREVIATION_TOKENS:
                print(tokens[i - 1])
                print(tokens[i])
                tokens[i] = ABBREVIATION_TOKENS[tokens[i]]["replacement"]
    return tokens
