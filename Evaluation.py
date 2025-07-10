
import numpy as np

from Ranking import Ranking
from main import ALPHA, MIN_SCORE, MODE, PARAM, SEPARATE_MODE, SEPARATE_PARAM, nDCG_DEPTH


class Evaluate():
    
    def __init__(self):
        pass

    def dcg(self, scores):

        return np.sum([
            (2**rel - 1) / np.log2(rank + 2)  # rank starts at 0
            for rank, rel in enumerate(scores)
        ])

    def ndcg(self, relevance_scores, ideal_scores):
        dcg_val = self.dcg(relevance_scores)
        idcg_val = self.dcg(ideal_scores)
        return dcg_val / idcg_val if idcg_val > 0 else 0.0

    def calculate_MAP(self, predicted_ids, relevant_ids):
        num_relevant = 0
        precisions = []
        for index, predicted_doc in enumerate(predicted_ids, start = 1):
            if predicted_doc in relevant_ids:
                num_relevant += 1
                precisions.append(float(num_relevant) / float(index))
        if precisions:
            ap = sum(precisions) / len(relevant_ids)
        else:
            ap = 0.0
        
        return ap


    def evaluate_with_relevance(self, df_queries, df_data):
        macro_ndcg_scores = []
        macro_precisions = []
        macro_recalls = []
        macro_f1s = []
        average_precisions = []

        all_pred_bin = []
        all_true_bin = []

        df_queries = df_queries["queries"]

        ranking = Ranking(df_data)
        print("Done with TFIDF computation")

        for row in df_queries:
            query = row["q"]
            relevant_ids = [entry[0] for entry in row["r"] ] 
            ideal_scores= [entry[1] for entry in row["r"] ] 
    
            relevant_docs = dict(row["r"])

            predicted_doc, res_all = ranking.query(query, SEPARATE_MODE, 
                                            SEPARATE_PARAM,
                                            MODE,
                                            PARAM,
                                            ALPHA,
                                            MIN_SCORE)
            
            predicted_ids = predicted_doc["official_id"]
            # Get predicted document ids from system

            predicted_set = set(predicted_ids)
            true_positives = len(predicted_set & set(relevant_ids))

            average_precisions.append(self.calculate_MAP(predicted_ids, relevant_ids))
            if len(predicted_ids) >= nDCG_DEPTH and len(ideal_scores) >= nDCG_DEPTH:
                relevance_scores_of_predicted_doc = [relevant_docs.get(doc_id, 0.0) for doc_id in predicted_ids[:nDCG_DEPTH]]
                ideal_scores_ndcg = sorted(ideal_scores, reverse = True)[:nDCG_DEPTH]
            elif len(ideal_scores) >= nDCG_DEPTH:
                relevance_scores_of_predicted_doc = [relevant_docs.get(doc_id, 0.0) for doc_id in predicted_ids] + [0] * (nDCG_DEPTH - len(predicted_ids))
                ideal_scores_ndcg = sorted(ideal_scores, reverse = True)[:nDCG_DEPTH]
            elif len(predicted_ids) >= nDCG_DEPTH:
                relevance_scores_of_predicted_doc = [relevant_docs.get(doc_id, 0.0) for doc_id in predicted_ids[:nDCG_DEPTH]]
                ideal_scores_ndcg = sorted(ideal_scores, reverse = True) + [0] * (nDCG_DEPTH - len(ideal_scores))
            else:
                relevance_scores_of_predicted_doc = [relevant_docs.get(doc_id, 0.0) for doc_id in predicted_ids] + [0] * (nDCG_DEPTH - len(predicted_ids))
                ideal_scores_ndcg = sorted(ideal_scores, reverse = True) + [0] * (nDCG_DEPTH - len(ideal_scores))

            # nDCG for this query
            macro_ndcg_scores.append(self.ndcg(relevance_scores_of_predicted_doc, ideal_scores_ndcg))

            # Binary relevance for metrics
            predicted_set = set(predicted_ids)
            true_positives = len(predicted_set & set(relevant_ids))

            precision = true_positives / len(predicted_set) if predicted_set else 0
            recall = true_positives / len(relevant_ids) if relevant_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1s.append(f1)

            for pid in predicted_ids:
                all_pred_bin.append(pid in relevant_ids)
            for rid in relevant_ids:
                all_true_bin.append(rid in predicted_set)


            res_relev = res_all[res_all['official_id'].isin(relevant_ids)]
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(res_relev[["name", "score_sparse", "score_dense", "score"]])
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        macro = {
            "ndcg": np.mean(macro_ndcg_scores),
            "precision": np.mean(macro_precisions),
            "recall": np.mean(macro_recalls),
            "f1": np.mean(macro_f1s),
            "MAP": np.mean(average_precisions)
        }

        # Micro
        micro_precision = sum(all_pred_bin) / len(all_pred_bin) if all_pred_bin else 0
        micro_recall = sum(all_pred_bin) / len(all_true_bin) if all_true_bin else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        micro = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        }

        return macro, micro