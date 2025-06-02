# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: VAMSI PUTTEPU

*INTERN ID*: CT04DN135

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## 🧠 Task 2: Sentiment Analysis using TF-IDF and Logistic Regression – Detailed Description

In the second task of the CODTECH Machine Learning Internship, we developed a **Sentiment Analysis** model using **TF-IDF vectorization** and **Logistic Regression**. This project introduced fundamental techniques in **Natural Language Processing (NLP)** and helped build a practical machine learning pipeline that can classify text-based sentiments, such as customer reviews.

We used a dataset named `data.csv`, which consisted of customer reviews labeled as either **positive** or **negative** sentiments. The main objective was to build a model that can accurately determine the sentiment of a given review. The deliverable was a Jupyter notebook demonstrating the full process from data preprocessing to model evaluation.

### 📁 Dataset Overview

The dataset contained two essential columns:
- `Review`: The actual customer review text
- `Sentiment`: The label indicating sentiment (1 for positive, 0 for negative)

Before model training, the text data required thorough preprocessing because raw text data is unstructured and contains noise, such as punctuation, numbers, and stopwords. These elements do not contribute to sentiment and can degrade model performance if not removed.

### 🔄 Text Preprocessing

To clean the data, we performed the following steps:
- Converted all text to lowercase
- Removed numerical characters
- Stripped punctuation using the `string` module
- Removed common stopwords using the NLTK library (`stopwords.words('english')`)

These steps help reduce dimensionality and improve the effectiveness of the vectorizer. We used a helper function `clean_text()` to apply these transformations consistently across the dataset.

### 🧮 TF-IDF Vectorization

Once cleaned, the text was converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization. This technique helps quantify the importance of words in a document relative to the corpus. Using `TfidfVectorizer` from `scikit-learn`, we converted all cleaned reviews into feature vectors suitable for machine learning.

The TF-IDF matrix was limited to a maximum of 5000 features to avoid overfitting and reduce training time. This vectorized representation was then used as input to our classifier.

### 🧠 Model Training

For classification, we used **Logistic Regression**, a simple yet powerful linear model that performs well on text data. We split the dataset into training and testing sets using an 80/20 ratio. The model was trained on the training data and then evaluated on the test set.

### 📊 Model Evaluation

We evaluated the model using standard classification metrics:
- **Accuracy**: The proportion of correct predictions
- **Confusion Matrix**: A breakdown of predicted vs actual labels
- **Classification Report**: Detailed metrics including precision, recall, and F1-score

The model achieved high accuracy, demonstrating that it successfully captured patterns in the text indicative of positive or negative sentiment.

### 🔮 Sample Predictions

We also included functionality to predict the sentiment of new, unseen reviews. The review is first cleaned using the same `clean_text()` method, then vectorized using the existing TF-IDF model, and finally classified using the trained logistic regression model.

### 🗂 Project Structure

The notebook clearly separates each step and includes outputs at each stage for clarity. Additional helper files such as `README.md` and `requirements.txt` ensure easy setup and reproducibility.

### ✅ Conclusion

This task was a hands-on introduction to natural language processing and text classification. By applying TF-IDF and logistic regression, we built a complete pipeline from raw text to sentiment prediction. The task reinforces essential ML concepts like text vectorization, feature engineering, and model evaluation. This foundational knowledge will be useful in future projects involving NLP, such as chatbot development, document classification, and more.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/a91823da-8a49-409c-bcd6-c38ba460fdba)
