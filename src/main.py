from preprocessing.reader import load
from preprocessing.embedding import embed


if __name__ == "__main__":
    questions, answers = load("../data/test.txt")
    embedding = embed(questions)
    
    print(questions[0], answers[0], embedding[0])