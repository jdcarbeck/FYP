from gensim.models.ldamodel import LdaModel
from .Corpus import Corpus
from .ConceptExtract import Concepts
import math
import statistics
import numpy as np

class SentenceRanker:
    def __init__(self, model: LdaModel, docs: Corpus, topic_dist):
        self.model = model
        self.docs = docs
        self.topic_dist = topic_dist
        self.max_topic = max(topic_dist[0], key=lambda x:x[1])

    def get_best(self, amount, method):
        # Compute Max Score
        # Sum Score
        # Max TF score
        # Sum Tf score
        pass

    """A sentence is made of a list of concepts"""
    def score_sentences(self, sentences: [str]):
        feq_concepts = {}
        for sent in sentences:
            concepts_in_sent = self.docs.sen2con[sent]
            for concept in concepts_in_sent:
                if concept in feq_concepts:
                    feq_concepts[concept] += 1
                else:
                    feq_concepts[concept] = 1
        num_of_concepts = len(feq_concepts.keys())
        sentences_scores = []
        for sent in sentences:
            concept_scores = []
            for concept in self.docs.sen2con[sent]:
                con_freq = (feq_concepts[concept] / num_of_concepts)
                concept_scores.append(self.concept_score(concept, con_freq))
            if(concept_scores != []):
                sentences_scores.append(tuple((sent, self.mean_log(concept_scores))))
        return sentences_scores
        
    """Concept scoring, using model to find word relation topic"""
    def concept_score(self, concept: str, con_freq, coef=0.4):
        # Convert concept to ID value
        con_id = self.model.dct.token2id[concept]
        # Find word topic Score
        topic_con_vec = self.model.lda_model.get_term_topics(con_id)
        # set max topic concept score
        max_con_score = self.get_topic_score(self.max_topic[0], topic_con_vec)
        # determine sum score
        sum_con_score = self.get_sum_score(topic_con_vec)

        max_tf_con_score = (coef * con_freq) + ((1 - coef) * max_con_score)

        sum_tf_con_score = (coef * con_freq) + ((1 - coef) * sum_con_score)

        mean = statistics.mean([max_con_score, sum_con_score, max_tf_con_score, sum_tf_con_score])

        return mean
    
    def get_topic_score(self, topic_id, topic_dist):
        topic_score = np.float64(0)
        for topic in topic_dist:
            if topic[0] == topic_id:
                topic_score += topic[1]
        return topic_score

    """Summation of each of the product of a topic2doc and con2topic"""
    def get_sum_score(self, con_topic_dist):
        sum_score = np.float64(0)
        for topic in self.topic_dist[0]:
            con_topic_score = self.get_topic_score(topic[0], con_topic_dist)
            sum_score += (topic[1] * con_topic_score)
        return sum_score

    def mean_log(self, val_list):  
        log_vals = np.log(val_list)
        try:
            return statistics.mean(log_vals)
        except statistics.StatisticsError:
            print(log_vals)
        