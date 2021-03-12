from typing import List


TOKEN_CHAR_NUM = "number"
TOKEN_CHAR_MONEY = "money"
TOKEN_CHAR_MONTH = "month"
TOKEN_CHAR_PERCENTAGE = "percentage"
TOKEN_CHAR_QUOTE = "#QUOTE#"
TOKEN_CHAR_URL = "URL"
TOKEN_YEAR = "year"


"""
This rule set can be be passed into the tokenise method which will tokenise certain tokens dependant on these rules.
    
TOKENISE_QUOTES
    Tokenise quotes will convert: 
        What was the meaning of 'to kill a mocking bird' -> What was the meaning of #quote#.
    This may or may not be useful, testing will be important.
    
TOKENISE_NUMBERS
    This tag will convert 123 -> #NUM#
    This should be useful as normally, specific numbers are not important, but the fact that there is a number is.
    
TOKENISE_MONTH
    This tag will convert any token of a month into the token #MONTH#
    
TOKENISE_PERCENTAGES
    This tag will convert any numeric percentages '14%' to #PERCENT#
    
REMOVE_QUESTION_MARKS
    This tag will remove any question mark as they may not be useful.
    
TOKENISE_URLS:
    This tag will convert urls from www.example.com to just #URL#
    
TOKENISE_MONEY:
    This tag will allow money tokens to be converted from $12.23c to #MONEY#
    
TOKENISE_YEAR:
    This tag will convert 4 digits integers to a tag #YEAR#
    
TOKENISE_COMMA_SEPERATED_NUMBERS:
    Often large numbers are seperated by a comma, this function will merge those tokens in two one tag #NUM#
"""
DEFAULT_RULES = {
    "TOKENISE_QUOTES": False,
    "TOKENISE_NUMBERS": True,
    "TOKENISE_MONTH": True,
    "TOKENISE_PERCENTAGES": True,
    "REMOVE_QUESTION_MARKS": True,
    "TOKENISE_URLS": True,
    "TOKENISE_MONEY": True,
    "TOKENISE_YEAR": True,
    "TOKENISE_COMMA_SEPERATED_NUMBERS": True
}


# A token which exists within this list will be converted to TOKEN_CHAR_MONTH
months_tokens = {
    "jan", "janurary", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june", "jul", "july",
    "aug", "augest", "sep", "sept", "september", "oct", "october", "nov", "novemeber", "dec", "december"
}


def parse_tokens(tokens, rules: dict = None) -> List[str]:
    """
    Parse a list of given tokens and rules to remove excessive tokens.

    This function will parse and replace tokens with specific tokens which allow us to reduce the amount of dimensoins
    and unique words encoded.

    Args:
        tokens: List of all tokens in a sentence.
        rules: The rules to which tokens are parsed. A full list can be seen above which explains what the rules do.

    Returns:
        This function returns the parsed tokens which simplifies the tokens whilst retaining semantic meaning.
    """
    # Populate rules for tokenisation
    rules = fill_rules(rules)

    # Lower all chars
    # tokens = [token.lower() for token in tokens]

    if rules["TOKENISE_QUOTES"]:
        tokens = parse_quotes(tokens)

    parsed_tokens = []
    for token in tokens:
        if token == "?" and rules["REMOVE_QUESTION_MARKS"]:
            continue

        if token in months_tokens and rules["TOKENISE_MONTH"]:
            token = TOKEN_CHAR_MONTH

        if ".com" in token and rules["TOKENISE_URLS"]:
            token = TOKEN_CHAR_URL

        if token[0] in {"$", "£"} and rules["TOKENISE_MONEY"]:
            token = TOKEN_CHAR_MONEY

        if token[-1] == "%" and is_num(token[:-1]) and rules["TOKENISE_PERCENTAGES"]:
            token = TOKEN_CHAR_PERCENTAGE

        if is_num(token) and "." not in token and len(token) == 4 and rules["TOKENISE_YEAR"]:
            token = TOKEN_YEAR

        if is_num(token) and rules["TOKENISE_NUMBERS"]:
            token = TOKEN_CHAR_NUM

        # This is not toggle-able since '&' and 'and' have the same meaning, so no
        # meaning is lost, however fewer dimensions are required using this.
        if token == "&":
            token = "and"

        # Add the converted tokens to the list
        parsed_tokens.append(token)

    # Convert any num,num to just num. e.g. 1 , 000 -> num rather than num,num
    if rules["TOKENISE_COMMA_SEPERATED_NUMBERS"]:
        parsed_tokens = merge_comma_seperated_numers(parsed_tokens)

    return parsed_tokens


def fill_rules(rules: dict) -> dict:
    """
    Populate the rules dictionary so that it auto fills any empty rules.

    It is assumed that not all rules will be included in the rules dictionary, this function will populate any rules to
    ensure that all rules are included to allow for full customisation of the tokenisation.

    Args:
        rules: The given rules. If none then return the default rules list, else, any missing rules will be replaced
            by the default rules in default rules dict.

    Return:
        The full list of toggleable rules.
    """
    if rules is None:
        return DEFAULT_RULES

    for rule in DEFAULT_RULES:
        if rule not in rules:
            rules[rule] = DEFAULT_RULES[rule]
    return rules


def merge_comma_seperated_numers(tokens: List[str]) -> List[str]:
    """
    Some numbers like 1,000 are seperated out to 1, ,, 000. This function merges them into one token.

    This function iterates backwards through the list of tokens merging tokens which are seperated by a comma. This
    should help reduce the size of the data whilst retaining the same semantic informations.

    Args:
        tokens: The list of expansive tokens.

    Returns:
        The list of tokens with the comma seperated number concatinated.
    """
    if TOKEN_CHAR_NUM in tokens:
        for i in range(len(tokens) - 1, -1, -1):
            if i < len(tokens):
                if tokens[i] == TOKEN_CHAR_NUM and tokens[i - 1] == "," and \
                        tokens[i - 2] in {TOKEN_CHAR_MONEY, TOKEN_CHAR_NUM}:
                    del tokens[i]
                    del tokens[i - 1]

    return tokens


def is_num(x: str) -> bool:
    """
    Check if a string is a number.

    This function will check if string is a number, if so return True otherwise return False.

    Args:
        x: Input string.

    Returns:
        True if string is a number.
    """
    try:
        float(x)
        return True
    except:
        return False


def parse_quotes(tokens: List[str]) -> List[str]:
    """
    Parse quote marks from a token list.

    This function will search for any quotations within the list of tokens and replace them with #QUOTE#. Often the data
    within the quote is not useful for the question description, therefore it is benificial to remove the quotes and
    replace them with the #QUOTE# Token.

    Args:
        tokens: List of strings which make up the question.

    Return:
        List of tokens with all quotes removed.
    """
    if "``" in tokens:
        # Store the indicies of the quotations in here.
        quotes = []

        # Search through the list of tags which make up a quote.
        start = -1
        for i in range(len(tokens)):
            if tokens[i] == "``":
                start = i
            if start != -1 and tokens[i] == "''":
                quotes.append((start, i))
                start = -1

        # Remove quote in reverse order to ensure indicies remain intact.
        quotes.reverse()
        for quote in quotes:
            tokens[quote[0]] = TOKEN_CHAR_QUOTE
            del tokens[quote[0] + 1: quote[1] + 1]

    return tokens
