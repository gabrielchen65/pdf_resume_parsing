import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise

def cosine_similarity(document1: str, document2: str) -> float:
    """Calculate the cosine similarity between two documents.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The cosine similarity between the two documents.
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the documents into TF-IDF vectors
    vectors = vectorizer.fit_transform([document1, document2])

    cosine_similarity_score = pairwise.cosine_similarity(vectors[0], vectors[1])
    # Calculate the cosine similarity between the two vectors
    # cosine_similarity = np.dot(vectors[0], vectors[1].T) / (np.linalg.norm(vectors[0].toarray()) * np.linalg.norm(vectors[1].toarray()))

    return cosine_similarity_score.item()

def evaluate(generated_dict, method="cosine_similarity"):
    generated_str = json.dumps(generated_dict)
    with open("./data/my_resume_ground_truth.json") as f:
        ground_truth_dict = json.load(f)
        ground_truth_str = json.dumps(ground_truth_dict)
    if method == "cosine_similarity":
        score = cosine_similarity(generated_str, ground_truth_str)
        print("Top level score: ", score)
        for key in ground_truth_dict:
            if key in generated_dict and len(generated_dict[key]) > 0:
                score = cosine_similarity(json.dumps(generated_dict[key]), json.dumps(ground_truth_dict[key]))
                print(f"{key} score: ", score)
            else:
                print(f"{key} not found in generated_dict")
    print("Evaluation complete.")