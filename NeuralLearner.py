import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import datasets
import wget

from LLM_simple import LLM
from main import Ranking, VERSION



from collections import defaultdict
from typing import Tuple
from pathlib import Path

from datasets import Dataset, DatasetDict
import pandas as pd

RANKING_MODE = "top"
RANKING_PARAM = 10
MIN_SCORE = 0

def loadWikirQueries(wikir_path: Path, split: str) -> Dataset:
    split_path = wikir_path / split
    if not split_path.is_dir():
        raise ValueError(f"Split {split} not found in {wikir_path}.")

    queries = pd.read_csv(split_path / "queries.csv")
    qrels   = pd.read_csv(split_path / "qrels", sep="\t", header=None)
    qrels.columns = ["id_left", "number", "id_right", "relevance"]
    qrels = qrels.merge(queries, on="id_left")
    qrels = qrels.rename(columns={
        "id_left": "query_id",
        "id_right": "doc_id",
        "text_left": "query"
    })
    qrels = qrels.drop(columns=["number", "query_id"])

    return Dataset.from_pandas(qrels, preserve_index=False)


def loadWikir(wikir_path: Path) -> Tuple[Dataset,DatasetDict]:
    queries_train = loadWikirQueries(wikir_path, "training")
    queries_valid = loadWikirQueries(wikir_path, "validation")
    queries_test  = loadWikirQueries(wikir_path, "test")

    documents = pd.read_csv(wikir_path / "documents.csv")
    documents = documents.rename(columns={
        "id_right": "doc_id",
        "text_right": "doc_text"
    })
    return Dataset.from_pandas(documents), DatasetDict({
             "train": queries_train,
        "validation": queries_valid,
              "test": queries_test
    })


def queryDatasetToQueryJson(queries: Dataset) -> dict:
    queries_to_documents = defaultdict(list)
    for example in queries:
        q = example["query"]
        d = example["doc_id"]
        r = example["relevance"]
        queries_to_documents[q].append([d,r])

    return {"queries": [{"q": query, "r": documents} for query, documents in queries_to_documents.items()]}


data_path = Path("wikIR1k")
documents, queries = loadWikir(data_path)
print(documents)
print(queries)

document_id_to_idx = {d["doc_id"]: idx for idx, d in enumerate(documents)}
print(queryDatasetToQueryJson(queries["train"])["queries"][10])
print(documents[document_id_to_idx[104206]])

NL_EMBEDDINGS_WIKI = f"storage/NL_embeddings_wiki_{VERSION}.npy"
NL_EMBEDDINGS_RECIPE = f"storage/NL_embeddings_recipe_{VERSION}.npy"

class NeuralLearner:

    def __init__(self, dataset, queries, recipes = False):
        

        if not os.path.exists(os.path.dirname( NL_EMBEDDINGS_WIKI)):
            os.mkdir(os.path.dirname( NL_EMBEDDINGS_WIKI))

        if not os.path.exists(os.path.dirname( NL_EMBEDDINGS_RECIPE)):
            os.mkdir(os.path.dirname( NL_EMBEDDINGS_RECIPE))
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device = "cuda")

        self.queries = queries
        self.queries_by_id = {
            str(idx): item["q"]
            for idx, item in enumerate(self.queries["queries"], start=1)
        }
        self.relevant_docs_by_id = {
            str(idx): {str(p[0]) for p in item['r']}
            for idx, item in enumerate(self.queries["queries"], start=1)
        }

        if recipes:
            self.documents_pandas = dataset.to_pandas()
            self.documents = dataset.to_dict()
            self.documents_text_list = [name + " " + tags + " " + ingr + " " + step + " " + dscr for name, tags, ingr, step, dscr in zip(self.documents["name"] , self.documents['tags'] , self.documents['ingredients'], self.documents['steps'], self.documents['description'])]
            
            print("Encoding documents")
            if not os.path.exists(NL_EMBEDDINGS_RECIPE):
                self.embeddings = self.model.encode(self.documents_text_list, show_progress_bar = True)
                np.save(NL_EMBEDDINGS_RECIPE, self.embeddings)
                print("Done")
            else:
                print("Fetching embeddings")
                self.embeddings = np.load(NL_EMBEDDINGS_RECIPE)
                print("Done")
            self.docs_by_id = {
                str(id) : text
                for id, text in zip(self.documents["official_id"], self.documents_text_list)
            }
            
        else:
            self.documents_pandas = dataset.to_pandas()
            self.documents = dataset.to_dict()
            self.documents_text_list = self.documents["doc_text"]
            print("Encoding documents")
            if not os.path.exists(NL_EMBEDDINGS_WIKI):
                self.embeddings = self.model.encode(self.documents_text_list, show_progress_bar = True, convert_to_numpy=True, batch_size = 64)
                np.save(NL_EMBEDDINGS_WIKI, self.embeddings)
                print("Done")
            else:
                print("Fetching embeddings")
                self.embeddings = np.load(NL_EMBEDDINGS_WIKI)
                print("Done")

            self.docs_by_id = {
                str(id) : text
                for id, text in zip(self.documents["doc_id"], self.documents_text_list)
            }
            
    def test_missing_word(self, query, text):
        embedding_text = self.model.encode([text])
        embedding_query = self.model.encode([query])
        sim = self.model.similarity(embedding_text, embedding_query)
        print("Document with query word not in document")
        print(sim)
        return sim[0]
        

    def query(self, query):
        
        embedded_query = self.model.encode([query])
        sims = self.model.similarity(self.embeddings, embedded_query)
        sim_list = sims.flatten().numpy()
        idx = Ranking.rank(sim_list,RANKING_MODE, RANKING_PARAM, MIN_SCORE)
        

        res_all = self.documents_pandas.copy()
        res_all["score"] = sims

        res = res_all.iloc[idx].copy()
        res =  res.sort_values("score", ascending=False)
        return res

    def get_metrics(self):
        #uses top k
        print("Building evaluator")
        ir_evaluator = InformationRetrievalEvaluator(
            queries=self.queries_by_id,
            corpus=self.docs_by_id,
            relevant_docs=self.relevant_docs_by_id,
            show_progress_bar = True
        )
        print("Getting results")
        results = ir_evaluator(self.model)
        print("OUTPUT sentences learner model")
        print(results)
        return results
    
if __name__ == "__main__":
    queries_NL = queryDatasetToQueryJson(queries["test"])
    #WIKI_NL = NeuralLearner(documents, queries_NL)
    #WIKI_NL.get_metrics()


    if not os.path.exists("./irse_documents_2025_recipes.parquet"):
        wget.download("https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2025_recipes.parquet")
    dataset_recipe = datasets.load_dataset("parquet", data_files="./irse_documents_2025_recipes.parquet")['train']


    if not os.path.exists("./irse_queries_2025_recipes.json"):
        wget.download("https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2025_recipes.json")
    queries_recipe = json.load(open("./irse_queries_2025_recipes.json", "r"))

    recipe_NL = NeuralLearner(dataset_recipe, queries_recipe, recipes=True)
    #recipe_NL.get_metrics()

    #recipe_NL.test_missing_word(query, text)
    
    

    query = "What are common mexican dishes?"
    res = recipe_NL.query(query)

    LLM(query, res)
