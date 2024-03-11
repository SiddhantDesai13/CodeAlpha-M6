#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
import joblib
import fitz


# In[2]:


dataset=pd.read_csv("C:\Siddhant\Projects\website_classification.csv")
dataset.shape


# In[3]:


dataset.head()


# In[4]:


df = dataset[['website_url','cleaned_website_text','Category']].copy()
df.head()


# In[5]:


pd.DataFrame(df.Category.unique()).values


# Now we need to represent each category as a number, so as our predictive model can better understand the different categories.

# In[6]:


# Create a new column 'category_id' with encoded categories 
df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)

# New dataframe
df.head()


# In[7]:


null_values = df.isnull().sum()
print(null_values)


# In[8]:


category_id_df


# In[9]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

# We transform each cleaned_text into a vector
features = tfidf.fit_transform(df.cleaned_website_text).toarray()

labels = df.category_id

print("Each of the %d text is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


# Spliting the data into train and test sets
# The original data was divided into features (X) and target (y), which were then splitted into train (75%) and test (25%) sets. Thus, the algorithms would be trained on one set of data and tested out on a completely different set of data (not seen before by the algorithm).

# In[10]:


X = df['cleaned_website_text'] # Collection of text
y = df['Category'] # Target or the labels we want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)


# In[11]:


y_train.value_counts()


# In[12]:


y_test.value_counts()


# In[13]:


models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    GaussianNB()
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
cv_df


# In[14]:


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc


# In[15]:


X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
calibrated_svc = CalibratedClassifierCV(estimator=model,
                                        cv="prefit")

calibrated_svc.fit(X_train,y_train)
predicted = calibrated_svc.predict(X_test)
print(metrics.accuracy_score(y_test, predicted))


# In[16]:


# Classification report
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_test,predicted,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],target_names= df['Category'].unique()))


# In[17]:


conf_mat = confusion_matrix(y_test, predicted,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="OrRd", fmt='d',
            xticklabels=category_id_df.Category.values, 
            yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16);


# In[18]:


for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >0:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual],id_to_category[predicted],
                                                                   conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Category', 
                                                                'cleaned_website_text']])


# In[19]:


import joblib
X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'], 
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
m1=CalibratedClassifierCV(estimator=m,
                                        cv="prefit").fit(tfidf_vectorizer_vectors, y_train)
joblib.dump(fitted_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(m1, 'calibrated_model.joblib')


# In[20]:


from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
import joblib
class ScrapTool:
    def visit_url(self, website_url):
        '''
        Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
        '''
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        content = requests.get(website_url,timeout=60).content
        
        #lxml is apparently faster than other settings.
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "website_text": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+
                                                               self.get_text_content(soup)
        }
        
        #Convert to Series object and return
        return pd.Series(result)
    
    def get_website_name(self,website_url):
        '''
        Example: returns "google" from "www.google.com"
        '''
        return "".join(urlparse(website_url).netloc.split(".")[-2])
    
    def get_html_title_tag(self,soup):
        '''Return the text content of <title> tag from a webpage'''
        return '. '.join(soup.title.contents)
    
    def get_html_meta_tags(self,soup):
        '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
        tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
        return ' '.join(content)
    
    def get_html_heading_tags(self,soup):
        '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
        tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)
    
    def get_text_content(self,soup):
        '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
        tags = soup.find_all(text=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                and isinstance(tag, bs4.element.Comment)==False\
                and not stripped_tag.isnumeric()\
                and len(stripped_tag)>0:
                result.append(stripped_tag)
        return ' '.join(result)
import spacy as sp
from collections import Counter
sp.prefer_gpu()
import en_core_web_sm
#anconda prompt ko run as adminstrator and copy paste this:python -m spacy download en
nlp = en_core_web_sm.load()
import re
def clean_text(doc):
    '''
    Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them
    '''
    doc = nlp(doc)
    tokens = []
    exclusion_list = ["nan"]
    for token in doc:
        if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum()==False) or token.text in exclusion_list :
            continue
        token = str(token.lemma_.lower().strip())
        tokens.append(token)
    return " ".join(tokens) 
text_c=clean_text("")
joblib.dump(text_c, 'text_c.joblib')


# In[ ]:


import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib as jl
from urllib.parse import urlparse
import fitz  # PyMuPDF for PDF processing

class TextClassifierGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Text Classifier")

        # Configure the style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#ececec')  # Frame color
        self.style.configure('TButton', background='#4caf50', foreground='#2e2e2e')  # Button color and darker text
        self.style.configure('TLabel', background='#ececec', foreground='#333333')  # Label color

        # Create a frame with the configured style
        self.frame = ttk.Frame(master, style='TFrame')
        self.frame.pack()

        self.label_url = ttk.Label(self.frame, text="Enter URL/choose a PDF file/ Pass the text:", style='TLabel')
        self.label_url.grid(row=0, column=0, padx=10, pady=10)

        self.entry_url = ttk.Entry(self.frame, width=80, font=('Arial', 16))
        self.entry_url.grid(row=0, column=1, padx=100, pady=100)

        self.browse_button = ttk.Button(self.frame, text="Browse PDF", command=self.browse_pdf)
        self.browse_button.grid(row=1, column=0, padx=10, pady=10,columnspan=3)

        self.predict_button = ttk.Button(self.frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.result_label = ttk.Label(self.frame, text="", style='TLabel')
        self.result_label.grid(row=3, column=0, columnspan=3, pady=100)

    def browse_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.entry_url.delete(0, tk.END)
            self.entry_url.insert(0, file_path)

    def predict(self):
        import spacy as sp
        from collections import Counter
        sp.prefer_gpu()
        import en_core_web_sm
        # anconda prompt ko run as administrator and copy paste this: python -m spacy download en
        nlp = en_core_web_sm.load()
        import re

        def clean_text(doc):
            '''
            Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them
            '''
            doc = nlp(doc)
            tokens = []
            exclusion_list = ["nan"]
            for token in doc:
                if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum() == False) or token.text in exclusion_list:
                    continue
                token = str(token.lemma_.lower().strip())
                tokens.append(token)
            return " ".join(tokens)

        input_text = self.entry_url.get()
        if input_text.lower().startswith("http"):
            # It's a URL, perform web scraping or use an API to fetch the content
            # Implement your web scraping logic here
            from bs4 import BeautifulSoup
            import bs4 as bs4
            from urllib.parse import urlparse
            import requests
            from collections import Counter
            import pandas as pd
            import os
            import joblib

            class ScrapTool:
                def visit_url(self, website_url):
                    '''
                    Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
                    '''
                    # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
                    content = requests.get(website_url, timeout=60).content

                    # lxml is apparently faster than other settings.
                    soup = BeautifulSoup(content, "lxml")
                    result = {
                        "website_url": website_url,
                        "website_name": self.get_website_name(website_url),
                        "website_text": self.get_html_title_tag(soup) + self.get_html_meta_tags(
                            soup) + self.get_html_heading_tags(soup) +
                                        self.get_text_content(soup)
                    }

                    # Convert to Series object and return
                    return pd.Series(result)

                def get_website_name(self, website_url):
                    '''
                    Example: returns "google" from "www.google.com"
                    '''
                    return "".join(urlparse(website_url).netloc.split(".")[-2])

                def get_html_title_tag(self, soup):
                    '''Return the text content of <title> tag from a webpage'''
                    return '. '.join(soup.title.contents)

                def get_html_meta_tags(self, soup):
                    '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
                    tags = soup.find_all(
                        lambda tag: (tag.name == "meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
                    content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords', 'description']]
                    return ' '.join(content)

                def get_html_heading_tags(self, soup):
                    '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
                    tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                    content = [" ".join(tag.stripped_strings) for tag in tags]
                    return ' '.join(content)

                def get_text_content(self, soup):
                    '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
                    tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]', "h1", "h2", "h3", "h4",
                                      "h5", "h6", "noscript"]
                    tags = soup.find_all(text=True)
                    result = []
                    for tag in tags:
                        stripped_tag = tag.strip()
                        if tag.parent.name not in tags_to_ignore \
                                and isinstance(tag, bs4.element.Comment) == False \
                                and not stripped_tag.isnumeric() \
                                and len(stripped_tag) > 0:
                            result.append(stripped_tag)
                    return ' '.join(result)

            scrapTool = ScrapTool()
            website = input_text
            try:
                web = dict(scrapTool.visit_url(website))
                text = (clean_text(web['website_text']))

            except:
                print("others")

        elif input_text.lower().endswith(".pdf"):
            import fitz  # PyMuPDF

            def extract_text_from_pdf(pdf_path):
                import fitz
                text = ""
                with fitz.open(pdf_path) as pdf_document:
                    # Iterate through all pages in the PDF
                    for page_number in range(pdf_document.page_count):
                        page = pdf_document[page_number]
                        text += page.get_text()
                return text

            p = extract_text_from_pdf(input_text)
            text = (clean_text(p))
        else:
            # It's assumed to be direct text input
            text_to_predict = input_text
            text = (clean_text(input_text))

        # Load the pre-trained model and vectorizer
        model = jl.load('calibrated_model.joblib')
        vectorizer = jl.load('tfidf_vectorizer.joblib')

        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])

        # Make predictions
        predicted_class = id_to_category[model.predict(text_vectorized)[0]]

        # Update the result label
        self.result_label.config(text=f"Your Article is from {predicted_class}")

# Create an instance of Tkinter
root = tk.Tk()

# Create an instance of the TextClassifierGUI class
text_classifier_gui = TextClassifierGUI(root)

# Run the Tkinter event loop
root.mainloop()


# In[ ]:


data_to_save = {
    'website_url': df['website_url'],
    'cleaned_website_text': df['cleaned_website_text'],
    'Category': df['Category'],
    'category_id': df['category_id']
}

# Create DataFrame
structured_df = pd.DataFrame(data_to_save)

# Save DataFrame to CSV
structured_df.to_csv('structured_data.csv', index=False)

# Save mean accuracy and standard deviation
acc.to_csv('model_accuracy.csv')

# Save category mapping dictionaries
pd.DataFrame.from_dict(category_to_id, orient='index', columns=['category_id']).to_csv('category_to_id.csv')
pd.DataFrame.from_dict(id_to_category, orient='index', columns=['Category']).to_csv('id_to_category.csv')


# In[ ]:




