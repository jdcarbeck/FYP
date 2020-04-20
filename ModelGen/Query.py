from .Corpus import Corpus
from .Model import Model
from .ConceptExtract import Concepts
from gensim.matutils import cossim

class Query:
    def __init__(self, corpus: Corpus, model: Model):
        self.corpus = corpus
        self.model = model


    """
        Based on a given document returns the most salient concepts in that document
    """
    def top_concepts(self, doc, keywords=3):
        concepts = Concepts(doc).get()
        print("\033[33mConcepts Extracted:\033[0m", concepts, "\n")
        doc_dist = self.model.topic_dist(concepts, show=False)
        concept_sim = []
       
        top_words = {}

        for topic_id, topic_value in doc_dist:
            top_n_words = self.model.lda_model.get_topic_terms(topic_id, topn=4)
            for word_id, word_prob in top_n_words:
                word_prob = topic_value * word_prob
                if word_id in top_words:
                    if word_prob > top_words[word_id]:
                        top_words[word_id] = word_prob
                else:
                    top_words[word_id] = word_prob
        top_n_words = list(top_words.items())
        top_n_words.sort(key=lambda tup: tup[1])

        keyword_list = []
        for word_id, value in top_n_words:
            word = self.model.dct.id2token[word_id]
            if word not in concepts and len(keyword_list) < keywords:
                keyword_list.append(word)   

        return keyword_list




    """
        Get the documents associated with a list of concepts
        gen a set of keywords to be associated with the concept chain
            • get documents (sentences) that relate to a given concept
            • combine into a document to use on the model
            • produce keywords specified that relate to given concepts
    """
    def get_concept_chain(self, concepts: [str], keywords=10):
        # find sents relating to given concepts
        unseen_sent = []
        for con in concepts:
            sentences = self.corpus.con2sen[con]
            for sent in sentences:
                if sent not in unseen_sent:
                    unseen_sent.append(sent)
        unseen_doc = " ".join(unseen_sent)
        
        unseen_concepts = [] 
        for sent in unseen_sent:
            unseen_concepts.append(Concepts(sent).get())
        unseen_model = Model(unseen_concepts, topics=10)
        join_concepts = " ".join(concepts)
        dist = unseen_model.topic_dist(Concepts(join_concepts).get())

        top_words = {}
        topic_prob = dist[0]
        word_probs = []

        for topic_id, topic_value in dist:
            top_n_words = unseen_model.lda_model.get_topic_terms(topic_id, topn=4)
            for word_id, word_prob in top_n_words:
                word_prob = topic_value * word_prob
                if word_id in top_words:
                    if word_prob > top_words[word_id]:
                        top_words[word_id] = word_prob
                else:
                    top_words[word_id] = word_prob
        top_n_words = list(top_words.items())
        top_n_words.sort(key=lambda tup: tup[1])

        cross_chain_query = []
        for word_id, value in top_n_words:
            word = unseen_model.dct.id2token[word_id]
            if word not in concepts and len(cross_chain_query) < keywords:
                cross_chain_query.append(word)   

        print("\033[34mExtended Query:\033[0m", (concepts + cross_chain_query), "\n")

        query = " ".join(list(concepts + cross_chain_query))
        query_concepts = Concepts(query).get()
        dist = self.model.topic_dist(query_concepts)

        return dist
        # print(dist[2])

    def retrieve_docs(self, concepts: [str], similarity = 0.80, query_len=10):
        topic_dist = self.get_concept_chain(concepts, keywords=query_len)
        similar_docs = []
        count = 0
        for doc in self.corpus.docs:
            doc_dist = self.model.topic_dist(doc.concepts)
            sim = cossim(doc_dist, topic_dist)
            if sim > similarity:
                count+=1
                doc_sen = list(doc.sen2con.keys())
                for sen in doc_sen:
                    similar_docs.append(sen)
        print("{}%% of documents found similar\n".format(count/len(self.corpus.docs)))
        return similar_docs

