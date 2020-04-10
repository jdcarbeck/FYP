from .Corpus import Corpus
from .Model import Model
from .ConceptExtract import Concepts
from gensim.matutils import cossim

class Query:
    def __init__(self, corpus: Corpus, model: Model):
        self.corpus = corpus
        self.model = model

    """
        Get the documents associated with a list of concepts
        gen a set of keywords to be associated with the concept chain
            • get documents (sentences) that relate to a given concept
            • combine into a document to use on the model
            • produce keywords specified that relate to given concepts
    """
    def get_concept_chain(self, concepts: [str], keywords=10):
        print("Original Query: ", concepts)

        # find sents relating to given concepts
        unseen_sent = []
        for con in concepts:
            sentences = self.corpus.con2sen[con]
            for sent in sentences:
                if sent not in unseen_sent:
                    unseen_sent.append(sent)
        unseen_doc = " ".join(unseen_sent)
        # concept extract the unseen_doc
        unseen_concepts = [] 
        for sent in unseen_sent:
            unseen_concepts.append(Concepts(sent).get())
        # print(unseen_doc)
        # print(unseen_concepts)
        unseen_model = Model(unseen_concepts, topics=10)
        # 0th element has the topic break down of document
        # 1st elemtn has the word to topic relation
        # 2nd element has the word to topic breakdown
        # unseen_model.print_model()
        join_concepts = " ".join(concepts)
        dist = unseen_model.topic_dist(Concepts(join_concepts).get())
        # for topic in unseen_model.lda_model.num_topics:
        #     top_topics = unseen_model.lda_model.get_topic_terms(topic)
        #     print(top_topics)
        # tuple(id,termid,value)

        top_words = {}
        topic_prob = dist[0]
        word_probs = []

        for topic_id, topic_value in topic_prob:
            # word (id, prob)
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
                   # Now sort the the top_words

        print("Extended Query: ", (concepts + cross_chain_query))



        query = " ".join(list(concepts + cross_chain_query))
        query_concepts = Concepts(query).get()
        dist = self.model.topic_dist(query_concepts)

        return dist
        # print(dist[2])

    def retrieve_docs(self, concepts: [str], similarity = 0.70):
        # topic_dist = self.model.topic_dist(concepts)
        topic_dist = self.get_concept_chain(concepts)
        similar_docs = []
        count = 0
        for doc in self.corpus.docs:
            for sent in doc.sen2con.keys():
                sent_concepts = doc.sen2con[sent]
                doc_dist = self.model.topic_dist(sent_concepts)
                sim = cossim(doc_dist[0], topic_dist[0])
                if sim > similarity:
                    count+=1
                    similar_docs.append(sent)
        print("{}%% of documents found similar".format(count/len(self.corpus.sen2con.keys())))
        return similar_docs, topic_dist

