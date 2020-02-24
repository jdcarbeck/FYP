import codecs
from bs4 import BeautifulSoup
from .ConceptExtract import Concepts
import re
import pickle
from nltk.tokenize import sent_tokenize

class Corpus:
    def __init__(self, filename, regen=False):
        self.docs = []
        self.concepts = []
        self.con2sen = {}
        if(regen):
            self.generate_docs(filename)
            self.gen_con2sen()
            with open('docs.pkl', 'wb') as f:
                pickle.dump(self.docs, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('concepts.pkl', 'wb') as f:
                pickle.dump(self.concepts, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('con2sen.pkl', 'wb') as f:
                pickle.dump(self.con2sen, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('docs.pkl', 'rb') as f:
                self.docs = pickle.load(f)
            with open('concepts.pkl', 'rb') as f:
                self.concepts = pickle.load(f)
            with open('con2sen.pkl', 'rb') as f:
                self.con2sen = pickle.load(f)

    def get_concepts(self):
        return self.concepts

    def generate_docs(self, filename):
        with codecs.open(filename, encoding='utf-8') as f:
            data = f.read()
        soup = BeautifulSoup(data, features="html.parser")
        documents = soup.find_all('doc')
        size = len(documents)
        print(("Processing: {} wiki articles").format(size))
        for doc in documents:
            title = doc.get('title')
            url = doc.get('url')
            uid = doc.get('id')
            text = doc.get_text()
            text = re.split(r'\n+', text)
            # text is now split by paragraph
            text = list(filter(None, text))
            for index, t in enumerate(text):
                d = Document(title, url, uid, index, t)
                self.docs.append(d)
                self.concepts.append(d.concepts)
        print("Finished!")

    """
        returns all sentences that contain a concept by searching each doc and returning that docs, check_sentences method
    """
    def gen_con2sen(self):
        for doc in self.docs:
            doc_con2sen = doc.con2sen
            for con in doc.concepts:
                if con in doc_con2sen:
                    if con in self.con2sen:
                        self.con2sen[con] = self.con2sen[con] + doc_con2sen[con]
                    else:
                        self.con2sen[con] = doc_con2sen[con]
                else:
                    self.con2sen[con] = []
                        

    def sents_with_con(self, concept):
        if concept in self.con2sen:
            return self.con2sen[concept]
        else:
            return []

class Document:
    def __init__(self, title, url, uid, paragraph, text):
        self.title = title
        self.url = url
        self.uid = uid
        self.text = text 
        self.paragraph = paragraph
        self.con2sen, self.concepts = self.gen_con2sen()

    """
    Takes a query term and searches if the term is in a sentence in the document
    """
    def gen_con2sen(self):
        con2sent = {}
        list_sent = sent_tokenize(self.text)
        for sent in list_sent:
            con_list = Concepts(sent).get()
            for con in con_list:
                if con in con2sent:
                    if sent not in con2sent[con]:
                        con2sent[con].append(sent)
                else:
                    con2sent[con] = [sent]
        return con2sent, list(con2sent.keys())

        
