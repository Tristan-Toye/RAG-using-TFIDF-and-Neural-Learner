import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import os
from collections import Counter, defaultdict
import copy 
from rake_nltk import Rake
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.matcher import PhraseMatcher
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from spacy.cli import download
import spacy
import re
import joblib
from scipy.sparse import save_npz, load_npz
from tqdm import tqdm
from joblib import Parallel, delayed
from datasketch import MinHash, MinHashLSH
from spacy.tokens import DocBin
from main import CLUSTER_N, DEDUPLICATE_PHRASES_THRESHOLD, INITIAL_WEIGTH_NAME, NORMALISE_TEXT, OUTPUT_DENSE_MATRIX_PATH, OUTPUT_LEMMA, OUTPUT_LSA_PIPELINE_PATH, OUTPUT_MAP_PATH, \
    OUTPUT_MATRIX_PATH, OUTPUT_PATTERNS, OUTPUT_PHRASES, OUTPUT_VECTOR_PATH, OUTPUT_DISH_NAMES, RESET_MAPPING, SAMPLE, SECOND_WEIGTH_NAME, SVD_COMPONENTS, print_example_doc



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



def extract_first_match(tags, category_set):
    tags_list = [tag.strip().lower() for tag in str(tags).split(',')]
    for tag in tags_list:
        if tag in category_set:
            return tag
    return 'unknown'



class TFIDF():
    
    
    def __init__(self, df):
        self.digits = re.compile(r'.*\d+.*')
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.df = df.copy()
        self.matcher = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        if not os.path.exists("storage"):
            os.mkdir("storage")

        if not os.path.exists(os.path.dirname( OUTPUT_MAP_PATH)):
            os.mkdir(os.path.dirname( OUTPUT_MAP_PATH))

        if not os.path.exists(os.path.dirname( OUTPUT_VECTOR_PATH)):
            os.mkdir(os.path.dirname( OUTPUT_VECTOR_PATH))

        if not os.path.exists(os.path.dirname( OUTPUT_MATRIX_PATH)):
            os.mkdir(os.path.dirname( OUTPUT_MATRIX_PATH))

        if not os.path.exists(os.path.dirname(OUTPUT_LEMMA)):
            os.mkdir(os.path.dirname(OUTPUT_LEMMA))

        if not os.path.exists(os.path.dirname(OUTPUT_PHRASES)):
            os.mkdir(os.path.dirname(OUTPUT_PHRASES))

        if not os.path.exists(os.path.dirname(OUTPUT_PATTERNS)):
            os.mkdir(os.path.dirname(OUTPUT_PATTERNS))

        if not os.path.exists(os.path.dirname(OUTPUT_DENSE_MATRIX_PATH)):
            os.mkdir(os.path.dirname(OUTPUT_DENSE_MATRIX_PATH))

        if not os.path.exists(os.path.dirname(OUTPUT_LSA_PIPELINE_PATH)):
            os.mkdir(os.path.dirname(OUTPUT_LSA_PIPELINE_PATH))

        if not os.path.exists(os.path.dirname(OUTPUT_DISH_NAMES)):
            os.mkdir(os.path.dirname(OUTPUT_DISH_NAMES))


    def TFIDF(self):

        if os.path.exists(OUTPUT_LEMMA) and os.path.exists(OUTPUT_MAP_PATH) and  os.path.exists(OUTPUT_VECTOR_PATH) and  os.path.exists(OUTPUT_MATRIX_PATH) \
              and os.path.exists(OUTPUT_DENSE_MATRIX_PATH) and os.path.exists(OUTPUT_LSA_PIPELINE_PATH):
            
            print("Fetching lemma dataframe")
            self.df = pd.read_parquet(OUTPUT_LEMMA)

         
            with open(OUTPUT_MAP_PATH, "r") as f:
                print("Fetching normalization map")
                normalization_map = json.load(f)
                print("Done")

            print("Fetching vectorizer")
            vectorizer = joblib.load(OUTPUT_VECTOR_PATH)
            print("Done")

            print("Fetching tfidf matrix")
            tfidf_matrix = load_npz(OUTPUT_MATRIX_PATH)
            print("Done")

            print("Fetching lsa pipeline")
            lsa_pipeline = joblib.load(OUTPUT_LSA_PIPELINE_PATH)
            print("Done")

            print("dense matrix")
            data = np.load(OUTPUT_DENSE_MATRIX_PATH)
            dense_matrix = data["dense_matrix"]
            print("Done")
            
            return normalization_map, vectorizer, tfidf_matrix, lsa_pipeline, dense_matrix, copy.deepcopy(self.df)
        else:
            
            print("Lemmatization")
            self._apply_lemmatization()
            print("Done")

            if os.path.exists(OUTPUT_MAP_PATH):
                with open(OUTPUT_MAP_PATH, "r") as f:
                    print("Fetching normalization map")
                    normalization_map = json.load(f)
                    print("Done")
            else:
                print("Extracting phrases")
                phrases = self._extract_phrases()
                print("Done")

                print("Cluster phrases")
                grouped_tags  = self._cluster_phrases(phrases)
                print("Done")

                print("Build normalization map")
                normalization_map = self._build_canonical_mapping(grouped_tags)
                print("Done")

            

            if 'normalized_text' not in self.df.columns or 'normalized_dscr' not in self.df.columns or RESET_MAPPING:
                print("Normalise text")
                self._prepare_relevant_text()
                self.df['normalized_text'] = self.normalise_input(self.df['lemmatized'].tolist(), normalization_map)
                print_example_doc(self.df)
                self.df.to_parquet(OUTPUT_LEMMA)
                print("Done")

            if not NORMALISE_TEXT:
                self.df['normalized_text'] = self.df['lemmatized']

            self._merge_name_with_normalized_text()

            print("vectorize text")
            print(self.df[self.df['official_id'] == 6453]['normalized_text'])
            matrix, vectorizer = self._vectorize_texts(self.df['normalized_text'])
            print("Done")

            if os.path.exists(OUTPUT_DENSE_MATRIX_PATH) and os.path.exists(OUTPUT_LSA_PIPELINE_PATH):
                print("Fetching lsa pipeline")
                lsa_pipeline = joblib.load(OUTPUT_LSA_PIPELINE_PATH)
                print("Done")

                print("dense matrix")
                data = np.load(OUTPUT_DENSE_MATRIX_PATH)
                dense_matrix = data["dense_matrix"]
                print("Done")
            else:

                #dense_matrix, dense_vectorizer = self._vectorize_texts_dense(self.df['lemmatized'])
                print("Compute dense represenation")
                lsa_pipeline, dense_matrix = self._compute_dense_representation(self.df['normalized_text'])
                print("Done")

            return normalization_map, vectorizer, matrix, lsa_pipeline, dense_matrix, copy.deepcopy(self.df)
            
            
    def _compute_dense_representation(self, texts):
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=36)
        normalizer = Normalizer(copy=False)
        dense_vectorizer = TfidfVectorizer(stop_words=list(self._clean_stop_words(STOP_WORDS)))
        lsa_pipeline = make_pipeline(dense_vectorizer, svd, normalizer)
        
        dense_matrix = lsa_pipeline.fit_transform(texts)
        np.savez_compressed(OUTPUT_DENSE_MATRIX_PATH, dense_matrix=dense_matrix)
        joblib.dump(lsa_pipeline, OUTPUT_LSA_PIPELINE_PATH)
        return lsa_pipeline, dense_matrix
    
    def _build_text(self, row):
        tags = " ".join(row['tags'])
        steps = row['steps']
        description = row['description'] or ""
        name = row['name'] or ""
        return name + " " + description + " " + steps

    def _prepare_relevant_text(self):
        self.df['lemmatized'] = self.df.progress_apply(self._build_text, axis=1)
        print_example_doc(self.df)


    def _merge_name_description(self, row):
        name = row['name'] or ""
  
        description = row['description'] or ""
        return " ".join(INITIAL_WEIGTH_NAME*[name]) + " " +  description 
    
    def _merge_name_normalized_text(self, row):
        name = row['name'] or ""

        normalized_text = row['normalized_text'] or ""
        return " ".join(SECOND_WEIGTH_NAME*[name]) + " " +  normalized_text

    def _merge_name_in_description(self):
        self.df['description'] = self.df.progress_apply(self._merge_name_description, axis=1)

    def _merge_name_with_normalized_text(self):
        self.df['normalized_text'] = self.df.progress_apply(self._merge_name_normalized_text, axis=1)

    def lemmatize_text(self,text):
        doc = self.nlp(text.lower())
        #return ' '.join([tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space and not self.digits.match(tok.lemma_)])
        return ' '.join([tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space and not tok.lemma_ in STOP_WORDS])
    
    def _apply_lemmatization_doc(self):
        self.df['lemmatized'] = self.df['text'].progress_apply(self.lemmatize_text)

    def _apply_lemmatization(self):
        if os.path.exists(OUTPUT_LEMMA):
            print("Found lemma dataframe")
            self.df = pd.read_parquet(OUTPUT_LEMMA)
        else:
            print("Running fast batch lemmatization...")
            self.df['course'] = self.df['tags'].apply(lambda tags: extract_first_match(tags, course_tags))

            self.df['name'] = self._batch_lemmatize(self.df['name'].tolist(), self.nlp, self.digits)
            self.df['tags'] = self._batch_lemmatize(self.df['tags'].tolist(), self.nlp, self.digits, keep_punctuation=True)
            self.df['ingredients'] = self._batch_lemmatize(self.df['ingredients'].tolist(), self.nlp, self.digits, keep_punctuation= True)
            self.df['description'] = self._batch_lemmatize(self.df['description'].tolist(), self.nlp, self.digits)
            self.df['steps'] =self._batch_lemmatize(self.df['steps'].tolist(), self.nlp, self.digits)
            
            self.df['tags'] = self.df['tags'].apply(lambda x: [tag.strip().lower() for tag in str(x).split(',')])
            self.df['ingredients'] = self.df['ingredients'].apply(lambda x: [tag.strip().lower() for tag in str(x).split(',')])
            
            
            self._merge_name_in_description()
            print("Preparing text")
            self._prepare_relevant_text()
            print("Done")

            self.df.to_parquet(OUTPUT_LEMMA)
    
    def _batch_lemmatize(self, texts, nlp, digits, keep_punctuation = False):
        lemmatized_texts = []
        # can also disable tagger
        for doc in tqdm(nlp.pipe(texts, batch_size=32, n_process = 10, disable=["ner", "parser"]), total=len(texts)):
            if keep_punctuation:
                lemmas = [
                    tok.lemma_.replace("-", " ").replace("_", " ").strip()
                    for tok in doc
                    if not tok.is_space and tok.lemma_ not in STOP_WORDS
                ]
            else:
                lemmas = [
                    tok.lemma_.replace("-", " ").replace("_", " ").strip()
                    for tok in doc
                    if not tok.is_punct and not tok.is_space and tok.lemma_ not in STOP_WORDS
                ]
            lemmatized_texts.append(" ".join(lemmas))
        return lemmatized_texts

    def _stratified_sample(self, df, group_col, n_samples):
        return df.groupby(group_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_samples), random_state=36)
        ).reset_index(drop=True)

    def _extract_phrases_rake(self, texts,):
        print("Extracting rake phrases")
        rake = Rake()
        phrases = []
        for txt in tqdm(texts, total = len(texts)):
            rake.extract_keywords_from_text(txt)
            phrases.extend(rake.get_ranked_phrases())
            #rake.clear() independant keyword extraction per doc
        print("Done")
        return phrases
    
    def _extract_phrases_spacy(self, texts, use_ner=True):
        print("Extracting spacy phrases")

        phrases = []
        for doc in tqdm(self.nlp.pipe(
            texts,
            batch_size=32,
            n_process=10,  
            disable=["tagger", "lemmatizer"]  # Keep parser + ner
        ), total = len(texts)):
            
            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if 3 < len(chunk.text)]
            phrases.extend(noun_chunks)

            
            if use_ner:
                entities = [ent.text.lower() for ent in doc.ents if 3 < len(ent.text)]
                phrases.extend(entities)
        print("Done")
        return phrases
    
    def _deduplicate_phrases(self, phrases, threshold=DEDUPLICATE_PHRASES_THRESHOLD):
        print("Deduplicate phrases with LSH")

        def get_minhash(text, num_perm=128):
            m = MinHash(num_perm=num_perm)
            for token in text.split():
                m.update(token.encode('utf8'))
            return m

        unique_phrases = []
        seen = set()
        lsh = MinHashLSH(threshold=threshold/100, num_perm=128)  

        minhashes = {}
        print("calculate minhash")
        for phrase in tqdm(phrases, desc="Building MinHashes", total= len(phrases)):
            mh = get_minhash(phrase)
            minhashes[phrase] = mh
            lsh.insert(phrase, mh)

        print("apply Locality Sensitive Hashing")
        for phrase in tqdm(phrases, desc="Filtering duplicates", total = len(phrases)):
            if phrase in seen:
                continue
            dupes = lsh.query(minhashes[phrase])
            unique_phrases.append(phrase)
            seen.update(dupes)

        print("Done")
        return unique_phrases

    def _deduplicate_phrases_fuzzy(self, phrases, threshold=DEDUPLICATE_PHRASES_THRESHOLD):
        print("Deduplicate phrases")
        phrases = sorted(set(phrases), key=len, reverse=True)
        unique_phrases = []

        def is_duplicate(phrase, existing):
            return any(fuzz.token_sort_ratio(phrase, e) >= threshold for e in existing)

        for phrase in tqdm(phrases, total = len(phrases)):
            if not Parallel(n_jobs=-1)(delayed(is_duplicate)(phrase, [e]) for e in unique_phrases):
                unique_phrases.append(phrase)

        print("Done")
        return unique_phrases

    def _extract_phrases(self):

   
        
        if os.path.exists(OUTPUT_PHRASES):
            with open(OUTPUT_PHRASES, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            df_sampled = self._stratified_sample(self.df, 'course', SAMPLE)
            texts = df_sampled['description']

        
            phrases_spacy = self._extract_phrases_spacy(texts)
            print(f"Length phrases spacy: {len(phrases_spacy)}")
            rake_phrases = self._extract_phrases_rake(texts)
            print(f"Length phrases rake: {len(rake_phrases)}")
            """
            phrases = phrases_spacy + rake_phrases
            print(f"Length phrases total: {len(phrases)}")
            phrases = self._deduplicate_phrases(phrases)
            print(f"Length phrases dedup: {len(phrases)}")
            """
            filtered_rake_phrases = [re.sub(r'\s+', " ", p.replace("_", " ").replace("-", " ").strip()) for p in rake_phrases if any(nc in p.lower() for nc in phrases_spacy)]
            print(f"Length phrases rake filtered: {len(filtered_rake_phrases)}")
            # Optionally add spaCy chunks directly if they're long enough
            extra_phrases = [re.sub(r'\s+', " ", nc.replace("_", " ").replace("-", " ").strip()) for nc in phrases_spacy if len(nc.replace("_"," ").split()) > 1]
            print(f"Length extra spacy phrases: {len(extra_phrases)}")
            all_ingredients = [
                re.sub(r'\s+', " ", ingredient.replace("_"," ").replace("-", " ").strip())
                for sublist in self.df["ingredients"]
                for ingredient in sublist
            ]

            all_tags = [
                re.sub(r'\s+', " ", tag.replace("_", " ").replace("-", " ").strip())
                for sublist in self.df["tags"]
                for tag in sublist
            ]
            phrases = filtered_rake_phrases + extra_phrases + all_ingredients + all_tags
            
            print(f"Length phrases : {len(phrases)}")
            with open(OUTPUT_PHRASES, "w") as f:
                json.dump(phrases, f, indent=2, ensure_ascii=False)
                print(f"Saved phrases → {OUTPUT_PHRASES}")
            
            return phrases
    
    def _cluster_phrases(self, phrases, n_clusters=CLUSTER_N):
        counts = Counter(phrases)
        print(f"Length phrases: {len(phrases)}")
        print(f"Length counts: {len(counts)}")
        filtered = [p for p, c in counts.items() if 3 < len(p) < 100 and c > 2]
        print(f"Length filtered phrases: {len(filtered)}")
        phrase_list = list(filtered)


        print("Get embeddings for clustering")
        embeddings = self.model.encode(
            phrase_list,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print("Clustering...")
        agglo = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',   # cosine similarity → distance = 1−cos
            linkage='average'    # average linkage works on arbitrary distances
        )
        labels = agglo.fit_predict(embeddings)

        """
        # build distance matrix
        print("Building distance matrix for clustering...")
        
        dist = [
            [100 - fuzz.ratio(a, b) for b in sorted_phrases]
            for a in tqdm(sorted_phrases, total = len(sorted_phrases))
        ]
       
        
       
        dist = pairwise_distances(
            sorted_phrases, sorted_phrases,
            metric=lambda x, y: 100 - fuzz.ratio(x, y),
            n_jobs=-1
        )

      
        

        print("Clustering phrases ...")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average' # minimizes variance
        )
        
        Distance Matrix vs. Raw Vectors
            We’re clustering on a precomputed distance matrix (100 – fuzz.token_sort_ratio), which is not guaranteed to satisfy the Euclidean metric assumptions that Ward linkage requires.

            Ward’s Requirement
            The Ward method minimizes total within‐cluster variance and assumes you’re clustering raw feature vectors in a Euclidean space. It uses the notion of “cluster centroids” and squared‐Euclidean distances to decide merges.

            Average (UPGMA) Flexibility
            Average linkage (a.k.a UPGMA) simply computes the average pairwise distance between all members of two clusters. It works fine with any symmetric distance matrix, even if it isn’t Euclidean. That makes it a safer, more general choice when your “points” are really just fuzzy‐distance scores.

        
        labels = clustering.fit_predict(dist) 

        
       
        With a defaultdict(list), you can do: grouped[label].append(phrase) without first checking if label not in grouped: grouped[label] = [].
        
        """
        
        print(f"Length labels: {len(labels)}")
        grouped = defaultdict(list)
        for i, lbl in enumerate(labels):
            grouped[f'group_{lbl}'].append(phrase_list[i])
        return grouped
        
    def _build_canonical_mapping(self,grouped_tags):
        mapping = {}
        for variants in grouped_tags.values():
            # pick the shortest, most central variant as canonical
            canonical = min(variants, key=lambda x: (len(x), variants.index(x)))
            for v in variants:
                mapping[v.lower()] = canonical.lower()
                
        with open(OUTPUT_MAP_PATH, "w") as f:
            json.dump(mapping, f, indent=2)
            print(f"Saved normalization map → {OUTPUT_MAP_PATH}")
        return mapping
    
    def _normalize_text(self, doc, matcher, mapping):
        norm = doc.text.lower()
        if NORMALISE_TEXT:
            
            for _, start, end in matcher(doc):
                span = doc[start:end].text.lower()
                tag  = mapping.get(span)
                if tag:
                    norm = norm.replace(span, tag)
        return norm

    def normalise_input(self, texts,  mapping):
        
        print("Building Matcher")
        if self.matcher is None:
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            if not os.path.exists(OUTPUT_PATTERNS):
                patterns = [self.nlp.make_doc(phrase) for phrase in tqdm(mapping.keys(), total = len(mapping.keys()))]
                doc_bin = DocBin(store_user_data=True) 
                for doc in patterns:
                    doc_bin.add(doc)
                doc_bin.to_disk(OUTPUT_PATTERNS)
            else:
                doc_bin = DocBin().from_disk(OUTPUT_PATTERNS)
                patterns = list(doc_bin.get_docs(self.nlp.vocab))

            self.matcher.add("TAGS", patterns)
            print("Done")
        #texts = self.df['lemmatized'].tolist()

        # Disable all pipeline components since text is already lemmatized



        normalized_texts = []
        print("Building normalised text")
        for doc in tqdm(self.nlp.pipe(texts, batch_size=32, n_process=10, disable=["tagger", "parser", "ner", "lemmatizer"]), total= len(texts)):
            normalized_texts.append(self._normalize_text(doc, self.matcher, mapping))
        print("Done")
        return normalized_texts
        #self.df['normalized_text'] = normalized_texts
        #print_example_doc(self.df)
    
    # Helper to match scikit-learn's token pattern
    def _clean_stop_words(self, stop_words):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return {w for sw in stop_words for w in token_pattern.findall(sw.lower())}
    
    def _vectorize_texts(self,texts):
        if os.path.exists(OUTPUT_VECTOR_PATH):
            print("Fetching vectorizer")
            vect = joblib.load(OUTPUT_VECTOR_PATH)
            print("Done")
        else:
            print("Building vectorizer ...")
            vect = TfidfVectorizer(stop_words=list(self._clean_stop_words(STOP_WORDS)))
            mat  = vect.fit_transform(texts)
            save_npz(OUTPUT_MATRIX_PATH, mat)
            print(f"Saved TF-IDF matrix     → {OUTPUT_MATRIX_PATH}")
            joblib.dump(vect, OUTPUT_VECTOR_PATH)
            print(f"Saved TF-IDF vectorizer → {OUTPUT_VECTOR_PATH}")

        if os.path.exists(OUTPUT_MATRIX_PATH):
            print("Fetching tfidf matrix")
            mat= load_npz(OUTPUT_MATRIX_PATH)
            print("Done")
        else:
            print("Building tfidf matrix")
            mat  = vect.fit_transform(texts)
            save_npz(OUTPUT_MATRIX_PATH, mat)
            print(f"Saved TF-IDF matrix     → {OUTPUT_MATRIX_PATH}")
        

        return mat, vect

    def _vectorize_texts_dense(self,texts):
 
        print("Building dense vectorizer ...")
        vect = TfidfVectorizer(stop_words=list(self._clean_stop_words(STOP_WORDS)))
        
        print("Building tfidf matrix for dense embeddings")
        mat  = vect.fit_transform(texts)

        

        return mat, vect