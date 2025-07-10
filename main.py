import json
import os
import datasets

import wget

import numpy as np
import pandas as pd
import json

import os

from tqdm import tqdm
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

tqdm.pandas()  # progress bar
import plotly.io as pio
# pick "browser", "iframe_connected", "png", etc., none of which require nbformat:
pio.renderers.default = "browser"
# show all columns
pd.set_option('display.max_columns', None)

# (optional) also show all rows
#pd.set_option('display.max_rows', None)

# (optional) widen the display so it won’t wrap:
pd.set_option('display.width', 200)



VERSION = 0.8

OUTPUT_PHRASES = f'storage/phrases/phrases_{VERSION}.json'
OUTPUT_LEMMA = f'storage/lemma/lemma_{VERSION}.parquet'
OUTPUT_MAP_PATH = f'storage/normalization_maps/map_{VERSION}.json'
OUTPUT_VECTOR_PATH = f'storage/vectorizers/vector_{VERSION}.joblib'
OUTPUT_MATRIX_PATH = f'storage/matrixs/matrix_{VERSION}.npz'
OUTPUT_DENSE_MATRIX_PATH = f'storage/dense_matrixs/dense_matrix_{VERSION}.npz'
OUTPUT_LSA_PIPELINE_PATH = f'storage/lsa_pipeline/lsa_pipeline_{VERSION}.joblib'
OUTPUT_PATTERNS = f'storage/patterns/patterns_{VERSION}.spacy'
OUTPUT_DISH_NAMES = f'storage/dishes/dishes_{VERSION}.json'

RESET_MAPPING = True



"""
VERSION = 0.4 -> 100 SVD embedding
VERSION = 0.5 -> no dscr + 250 SVD embedding
VERSION = 0.6 -> 0.5 with no normalization
VERSION = 0.7 -> 
VERSION = 0.8 -> normalised sims metrics and only name for sparse
"""

INITIAL_WEIGTH_NAME = 1
SECOND_WEIGTH_NAME = 1

SAMPLE_SIZE = 200
CLUSTER_N = 5000



nDCG_DEPTH = 10
DEDUPLICATE_PHRASES_THRESHOLD = 95

SEPARATE_MODE = "top"
SEPARATE_PARAM = 1
MODE = "statistical"
PARAM = 16
ALPHA = 0.85




MIN_SCORE = 0
SAMPLE = 200
NORMALISE_TEXT = True
SVD_COMPONENTS = 250
DUPLICATE_DISHES = 1


course_tags = {
    'main-dish', 'side-dishes', 'appetizers', 'desserts', 'breads',
    'soups-stews', 'salads', 'beverages', 'breakfast', 'lunch', 'snacks'
}

sample_queries = [
    "a cajun style gumbo with an easy roux",
    "I am feeling like eating shrimp tacos tonight. What's a good recipe?",
    "recipe for easy vegetarian lasagna",
    "How do I make spageti and meatballs?",
    "15 minute lunch recipe",
    "Give me suggestion for some easy vegetarian weeknight dinner recipes"
]

def print_example_doc(df):
    first_entry = df.iloc[0]

    for column, value in first_entry.items():
        print(f"\n{column.upper()}:")
        #print(textwrap.fill(str(value), width=80))
        print(value)


def extract_first_match(tags, category_set):
    tags_list = [tag.strip().lower() for tag in str(tags).split(',')]
    for tag in tags_list:
        if tag in category_set:
            return tag
    return 'unknown'




if __name__ == "__main__":
    if not os.path.exists("./irse_documents_2025_recipes.parquet"):
        wget.download("https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2025_recipes.parquet")
    dataset = datasets.load_dataset("parquet", data_files="./irse_documents_2025_recipes.parquet")['train']
    dataset = dataset.to_pandas()

    if not os.path.exists("./irse_queries_2025_recipes.json"):
        wget.download("https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2025_recipes.json")
    queries = json.load(open("./irse_queries_2025_recipes.json", "r"))

    print("One document:")
    print_example_doc(dataset)


    print("Ideal query responses:")
    print(queries["queries"][0])




    # TFIDF, Ranking, Evaluation

    print("#######################################################")
    print("#######################################################")
    print("#######################################################")

    #tfidf = TFIDF(dataset)
    #normalization_map, vectorizer, matrix, lsa_pipeline, dense_matrix, df = tfidf.TFIDF()
    sample_queries = [
        "a cajun style gumbo with an easy roux",
        "I am feeling like eating shrimp tacos tonight. What's a good recipe?",
        "recipe for easy vegetarian lasagna",
        "How do I make spageti and meatballs?",
        "15 minute lunch recipe",
        "Give me suggestion for some easy vegetarian weeknight dinner recipes"
    ]
    from Ranking import Ranking
    ranking = Ranking(dataset)
    res = ranking.query(sample_queries[0])
    # query = " "
    # LLM(query, res)
    #exit()
    from Evaluation import Evaluate
    #eval = Evaluate()
    #results = eval.evaluate_with_relevance({"queries": queries["queries"]}, dataset)
    #print(results)



