
import copy 
import numpy as np
import pandas as pd
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import plotly.graph_objects as go
from TFIDF import TFIDF
from main import ALPHA, DUPLICATE_DISHES, MIN_SCORE, MODE, PARAM, SEPARATE_MODE, SEPARATE_PARAM
from nltk.corpus import wordnet as wn


class Ranking():

    def __init__(self, df_raw: pd.DataFrame):

        self.nlp = spacy.load("en_core_web_sm")


        self.tfidf = TFIDF(df_raw)
        self.normalization_map , self.vectorizer , self.tfidf_matrix, self.lsa_pipeline, self.dense_matrix, self.df = self.tfidf.TFIDF()
        self.dish_names = self._get_dish_names()

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER", validate=True)
        self.patterns = [self.nlp.make_doc(name) for name in self.dish_names]
        self.matcher.add("DISH", self.patterns)


    def _duplicate_dishes_in_query(self, query: str, n: int = DUPLICATE_DISHES) -> str:
        """
        For any recipe name appearing in `query`, repeat it `n` times.
        """
        doc = self.nlp(query)
        matches = self.matcher(doc)
        # Sort matches by their start index
        matches = sorted(matches, key=lambda m: m[1])

        out = []
        last_end = 0
        for _, start, end in matches:
            # Append text before the match
            out.append(doc[last_end:start].text_with_ws)
            span = doc[start:end]
            # Append the matched span n times (preserving whitespace)
            out.append(" ".join(n * [span.text.strip()]))
            last_end = end
        # Append any trailing text
            out.append(" ")
        out.append(doc[last_end:].text_with_ws)

        # Join and strip to clean up leading/trailing spaces
        return "".join(out).strip()

    def _get_dish_names(self):

        

        # 1. Grab the master “food” synset
        food = wn.synset('food.n.02')

        # 2. Recursively collect all hyponyms (i.e. more specific foods)
        food_synsets = list(food.closure(lambda s: s.hyponyms()))

        # 3. Optionally include “food” itself
        food_synsets.append(food)

        # or just the words:
        food_words = [lemma.name().replace('_', ' ') 
                            for s in food_synsets 
                            for lemma in s.lemmas()]
        food_words.append('quesadilla')
        return food_words


    def _similarities_sparse(self, query):
        query_sparse = copy.deepcopy(query)
        query_sparse = self._preprocess_sparse_query(query_sparse)
        print("Query:")
        print(query_sparse)
        print("Tokenise")

        print(f"Query: {query_sparse}")
        print("quesadilla" in self.vectorizer.get_feature_names_out())

        vec_sparse = self.vectorizer.transform(query_sparse)
        print("Calculate similarity")
        print(f"Size tfidf matrix: {self.tfidf_matrix.shape}")
        sims = cosine_similarity(vec_sparse, self.tfidf_matrix).flatten()

        return sims
        #return sims / sims.max() if sims.max() > 0 else sims
    
    def _similarities_dense(self, query):
        query_dense = copy.deepcopy(query)
        query_dense = self._preprocess_dense_query(query_dense)
        print("Tokenise")
        q_vec = self.lsa_pipeline.transform([query_dense])
        print("Calculate similarity")
        print(f"Size tfidf matrix: {self.dense_matrix.shape}")
        sims = cosine_similarity(q_vec, self.dense_matrix).flatten()
        return sims
        #return sims / sims.max() if sims.max() > 0 else sims

        
    def compute_jaccard_score(self, doc_tokens: set, query_tokens: set) -> float:
  
        if not query_tokens:
            return 0.0
        inter = doc_tokens & query_tokens
        union = doc_tokens | query_tokens
        return len(inter) / len(union) if union else 0.0
        
    def query(self, query: str, seperate_mode:str = SEPARATE_MODE, seperate_param: str = SEPARATE_PARAM, \
                mode:str = MODE, param = PARAM, alpha: float = ALPHA, min_score:int  = MIN_SCORE):
         
        """
        Dynamic search based on cosine similarity with multiple cutoff strategies.
        
        Args:
            query: User query string.
            mode: One of "relative", "statistical", "absolute", or "derivative".
            param: Sensitivity parameter for the mode.
                relative: >= max * param
                statistical: >= mean + param *std
                absolute: >= param
                dervivative: find first occurance of diff > threshold or "auto" -> find max derivative absolute value
            min_score: Minimum number of results to return.

        Returns:
            DataFrame with matched recipes and scores.
        """

        assert(0 <= alpha <= 1)
        
        print("Compute TFIDF similarities")
        sims_sparse = self._similarities_sparse(query)
        print("Compute dense similarities")
        sims_dense = self._similarities_dense(query)


        print(f"length sparse sim: {len(sims_sparse)}")
        print(f"length dense sim: {len(sims_dense)}")
        print("Done similarity calculations")

        if seperate_param:
            sparse_clear_idx = self.rank(sims_sparse, seperate_mode,seperate_param, min_score)
            dense_clear_idx = self.rank(sims_dense, seperate_mode,seperate_param, min_score)

        sims = alpha*sims_sparse  + (1- alpha) * sims_dense

        sparse_dense_idx = self.rank(sims, mode, param, min_score)
        
        if seperate_param:
            combined = np.unique(
                np.concatenate((sparse_clear_idx, dense_clear_idx, sparse_dense_idx))
            )
        else:
            combined = sparse_dense_idx

        print(f"lenght combined: {len(combined)}, length sparse & dense{len(sparse_dense_idx)}")
        
        


        # Retrieve results
        
        res_all = self.df.copy()
        res_all["score_sparse"] = sims_sparse
        res_all["score_dense"] = sims_dense
        res_all["score"] = sims

        res = res_all.iloc[combined].copy()
        res =  res.sort_values("score", ascending=False)
        
        

        
        print(res[["name", "score_sparse", "score_dense", "score"]])

        self._plot_sorted_similarities(sims)
        return res, res_all




    def _plot_sorted_similarities(self, sims):
        """
        Plots a line chart of similarities sorted in descending order using Plotly,
        and fixes the y-axis to always show the full 0–1 range.
        
        Parameters:
            sims (array-like): 1D array of similarity scores.
        """
        # Csort descending


        sorted_sims = np.sort(sims)[::-1][:100]
        # Create Plotly figure
        fig = go.Figure(go.Scatter(
            y=sorted_sims,
            mode='lines',
            name='Similarity'
        ))
        
        # Update layout to fix y-axis from 0 to 1
        fig.update_layout(
            title="Similarity Scores (Sorted Highest to Lowest)",
            xaxis_title="Document Rank",
            yaxis_title="Cosine Similarity",
            yaxis=dict(range=[0, 0.6])  # always shows full possible similarity range
        )
        
        # Show the interactive plot
        fig.show()
        fig.write_html("similarity_plot.html", include_plotlyjs="cdn")

    @staticmethod
    def rank( sims, mode, param, min_score):
        if sims.size == 0:
            return None
        if mode == "top":
            thr = min(param, sims.size)
            sorted_ix = sims.argsort()[::-1]
            sel = sorted_ix[:thr]
            idx = sel[sims[sel] >= min_score]
        elif mode == "relative":
            thr = sims.max() * param
            idx = np.where(sims >= thr)[0]
        elif mode == "statistical":
            thr = sims.mean() + param * sims.std()
            idx = np.where(sims >= thr)[0]
        elif mode == "absolute":
            idx = np.where(sims >= param)[0]
        elif mode == "derivative":
            idx = Ranking.select_top_by_derivative(sims, threshold=param)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return idx

    @staticmethod
    def select_top_by_derivative( scores: np.ndarray, threshold="auto") -> np.ndarray:
        sorted_ix = scores.argsort()[::-1]
        sorted_scores = scores[sorted_ix]
        diffs = np.abs(np.diff(sorted_scores))

        if diffs.size == 0:
            cutoff = sorted_scores.size
        else:
            if threshold == "auto":
                drop_ix = np.argmax(diffs)
                cutoff = drop_ix + 1
            else:
                large_drops = np.where(diffs > threshold)[0]
                cutoff = (large_drops[0] + 1) if large_drops.size > 0 else sorted_scores.size

        sel = sorted_ix[:cutoff]
        return sel
        
    
    def _preprocess_sparse_query(self,query: str) -> str:
        lem = self.tfidf.lemmatize_text(query)
        q = self.tfidf.normalise_input([lem], self.normalization_map)
        tmp = self._duplicate_dishes_in_query(q[0])

        return [tmp]
    
    def _preprocess_dense_query(self,query: str) -> str:
        return self.tfidf.lemmatize_text(query)

    def _duplicate_entities(self, text: str, n: int = 2) -> str:
        """
        Reconstruct `text`, but each time an entity is encountered,
        repeat that entire entity span `n` times.
        """
        doc = self.nlp(text, disable = ["lemmatize"])
        # Map from token-index → the Span that starts there
        entity_starts = {ent.start: ent for ent in doc.ents}

        out_parts = []
        i = 0
        while i < len(doc):
            if i in entity_starts:
                ent = entity_starts[i]
                # emit the entity span `n` times
                for _ in range(n):
                    # .text gives the raw substring, and
                    # ent[-1].whitespace_ preserves whatever whitespace followed it
                    out_parts.append(ent.text)
                    out_parts.append(ent[-1].whitespace_)
                i = ent.end  # skip past this entity
            else:
                # non-entity token: just emit its text+whitespace
                out_parts.append(doc[i].text_with_ws)
                i += 1

        return "".join(out_parts)