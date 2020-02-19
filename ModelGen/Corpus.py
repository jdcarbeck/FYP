import codecs
from bs4 import BeautifulSoup
from .ConceptExtract import Concepts
import re

class Corpus:
    docs = []
    concepts = []

    def __init__(self, filename):
        self.docs = self.generate_docs(filename)

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

class Document:
    def __init__(self, title, url, uid, paragraph, text):
        self.title = title
        self.url = url
        self.uid = uid
        self.text = text
        self.paragraph = paragraph
        self.concepts = Concepts(text).get()