# part-3-nlp-sequence-modeling

## Task 1: Dataset Understanding

**1. Number of Records:**
The dataset contains a total of **1,500** records.

**2. Target Labels/Classes:**
The text is classified into three target sentiment labels:

* `neutral`
* `positive`
* `negative`

**3. Sample Text Records:**
Here are three sample records from the dataset showing the original data structure:

* **Record 1:** "I need information about the payment process. My ticket number is 78732. Please respond as soon as possible." *(Label: neutral, Channel: chat, Urgent: Yes)*
* **Record 2:** "I need information about the payment process." *(Label: neutral, Channel: phone, Urgent: No)*
* **Record 3:** "The refund process was fast and convenient. I appreciate the quick response." *(Label: positive, Channel: email, Urgent: No)*

**4. Average Text Length:**
The average length of the customer messages in the dataset is approximately **12.72 words**.

**5. Class Distribution:**
The sentiment classes are relatively well-balanced across the dataset:

* **Neutral:** 524 records (34.9%)
* **Negative:** 497 records (33.1%)
* **Positive:** 479 records (31.9%)

==========================================================

## Task 2: Text Preprocessing

Built the preprocessing pipeline for dataset and generated a cleaned, preprocessed file named `customer_support_preprocessed.csv`.

Here is the breakdown of how standard raw text is transformed into a format that sequence-based deep learning models can understand.

### 1. Cleaning the Text

First, we applied basic normalization to reduce the complexity of the text:

* **Lowercasing:** Everything was converted to lowercase (e.g., "Please" becomes "please").
* **Removing symbols:** Punctuation and numbers were stripped using regular expressions, keeping only alphabetical letters and spaces.

*Note on Stopwords: For sequence models (like RNNs, LSTMs, or Transformers), we usually **keep** stopwords (like "is", "the", "as"). These models rely on the sequential flow and context of the sentence, and removing words disrupts the grammatical structure.*

### 2. Tokenization

Machine learning models cannot read text; they only understand numbers. Tokenization maps each unique word in the dataset to a unique integer index (e.g., "i" = 5, "need" = 30).

* **Vocabulary Size:** Across the entire dataset, the tokenizer identified a vocabulary of **182 unique words**.

### 3. Padding and Truncating Sequences

Because neural networks require input arrays of a fixed size, but customer messages vary in length, we have to standardize them.

* We set a **maximum sequence length of 25** (safely above the average text length of 12.7 words we found in Task 1).
* **Padding:** Messages shorter than 25 words get padded with zeros (`0`) at the end.
* **Truncating:** Messages longer than 25 words would get cut off at the 25th word.

---

### End-to-End Example Transformation

Here is how the very first record in the dataset was transformed through this pipeline:

**1. Original Text:**

> "I need information about the payment process. My ticket number is 78732. Please respond as soon as possible."

**2. Cleaned Text:**

> "i need information about the payment process my ticket number is please respond as soon as possible" *(Notice the numbers and punctuation are gone).*

**3. Tokenized Sequence (Converted to integers):**

> `[5, 30, 137, 40, 2, 91, 34, 4, 7, 8, 3, 11, 13, 9, 14, 9, 15]`

**4. Padded Sequence (Fixed length of 25):**

> `[5, 30, 137, 40, 2, 91, 34, 4, 7, 8, 3, 11, 13, 9, 14, 9, 15, 0, 0, 0, 0, 0, 0, 0, 0]`

This fixed-length integer array is now ready to be fed into a deep learning model's Embedding layer.

==========================================================

## Task 3: Text Vectorization

Both **Bag of Words (BoW)** and **TF-IDF** (Term Frequency-Inverse Document Frequency) matrices using your preprocessed text.

Here are the results of the vectorization:

* **Bag of Words Matrix Shape:** `(1500, 180)` — This means we have 1,500 records, and each record is represented by an array of 180 numbers (representing the frequency of each of the 180 unique words in the vocabulary).
* **TF-IDF Matrix Shape:** `(1500, 180)` — Same dimensions, but the values inside are weighted based on how rare or common the word is across the entire dataset.

Here is an example of what the **TF-IDF** vector looks like for the first document (*"I need information about the payment process..."*). Only the non-zero scores (the words actually present in the text) are shown:

* `as`: 0.4553
* `information`: 0.3586
* `payment`: 0.3409
* `about`: 0.2960
* `process`: 0.2890
* `need`: 0.2875
* `the`: 0.1004

*Notice how a highly common word like "the" gets a very low score (0.1004), while more specific, meaningful words like "information" (0.3586) or "payment" (0.3409) receive higher importance scores!*

---

### Why must text be converted into vectors?

At their core, Machine Learning and Deep Learning models are built on mathematical formulas and linear algebra (like matrix multiplication and gradient descent).

**1. Computers only understand numbers:** A computer cannot natively understand the meaning, emotion, or syntax of the word "payment". It can only process numerical values. Vectorization translates human language into a mathematical language the computer can compute.
**2. Measuring "Distance" and "Similarity":** By converting text into multi-dimensional vectors, we place sentences into a mathematical space. Models can then calculate the "distance" between vectors. If two vectors are close together in that space, the model learns that the texts are similar in meaning or sentiment (e.g., "fast" and "quick" will have similar vector profiles).
**3. Weighting Importance:** Techniques like TF-IDF allow the model to distinguish between important words (like "refund" or "terrible") and unimportant words (like "the" or "is") before it even starts training, significantly improving the model's accuracy.

**Traditional vs. Sequence Approaches:**

* **Traditional (BoW / TF-IDF):** Treats text as a "bag" of words. It counts frequencies but completely ignores the *order* of words. (e.g., "Not good, very bad" and "Not bad, very good" might look identical to the model).
* **Sequence-based (Tokenization from Task 2 + Word Embeddings):** Converts sentences into ordered lists of integers `[5, 30, 137...]`. When paired with Deep Learning (like RNNs or Transformers), the model preserves the exact sequence of the words, allowing it to understand grammar, context, and complex negations.

=======================================================================

## Task 4: Baseline Model

For our baseline, I built a **Logistic Regression model paired with TF-IDF vectorization**. This is a highly standard, traditional Machine Learning approach for text classification.

Here are the results of the evaluation on the 20% hold-out test set:

**Baseline Model Accuracy:** **1.0000 (100%)**

**Classification Report:**

```text
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00       109
     neutral       1.00      1.00      1.00       104
    positive       1.00      1.00      1.00        87

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300

```

### Analysis of the Baseline Model

Interestingly, the model achieved **100% accuracy** across all metrics (Precision, Recall, and F1-score) for all three sentiment classes (`negative`, `neutral`, `positive`).

**Why did this happen?**
Achieving a perfect 1.0 score usually indicates one of a few things about the dataset:

1. **Highly distinct vocabulary:** The words used in positive, negative, and neutral tickets in this dataset are likely very distinct with almost zero overlap in how they signal sentiment (e.g., negative tickets always use words like "unhappy" or "failing", while positive tickets strictly use words like "fast", "great", "convenient").
2. **Synthetic or simplified data:** If the dataset was synthetically generated or heavily curated, it often lacks the nuanced, messy, or sarcastic language found in real-world human text, making it extremely easy for a basic linear model to draw clear boundaries.

### Metrics Explained

Even though the scores are perfect here, here is what these metrics tell us in a normal scenario:

* **Precision:** Out of all the messages the model *claimed* were "positive", how many actually were? (Avoids false positives).
* **Recall:** Out of all the *actual* "positive" messages in the test set, how many did the model successfully find? (Avoids false negatives).
* **F1-Score:** The harmonic mean of Precision and Recall. It gives a balanced view of the model's performance.

### Why set a Baseline?

We establish a baseline model using traditional methods (TF-IDF + Logistic Regression) so we have a benchmark to beat. It tells us the "minimum acceptable performance". Because our traditional model achieved 100%, a more complex deep learning model (like an LSTM or Transformer) won't actually yield higher accuracy here—but in complex, real-world, highly ambiguous datasets, traditional models usually plateau around 75-85%, which is where sequence models shine.

====================================================

## Task 5: Sequence Model or Architecture (LSTM)

I have successfully built and trained a sequence-based Deep Learning model using an **LSTM (Long Short-Term Memory)** network.

Just like our baseline model, the LSTM achieved **100% accuracy** on the test set. While the simple dataset didn't give the LSTM a chance to flex its advanced capabilities, understanding *how* this model works is critical for modern NLP (as it's the foundational concept leading up to architectures like Transformers/GPT).

Here is the step-by-step breakdown of the sequence model architecture we just built:

#### 1. Input Sequence

* **What it is:** The input to the model is the padded integer arrays we created in Task 2 (e.g., `[5, 30, 137, ... 0, 0]`).
* **Why it matters:** Unlike Bag-of-Words or TF-IDF, which jumble all the words together like a salad, this input maintains the exact sequence and position of every word in the sentence.

#### 2. Embedding Layer

* **What it does:** It takes the integer sequences and converts each integer into a dense vector of fixed size (we used `embedding_dim = 32`).
* **Why it matters:** Instead of treating words as isolated numbers, the Embedding layer learns to place similar words close to each other in a 32-dimensional mathematical space. For example, the model learns that "fast" and "quick" have similar meaning vectors, whereas traditional models treat them as completely unrelated features.

#### 3. Recurrent / Sequence Layer (LSTM)

* **What it does:** The LSTM (Long Short-Term Memory) layer processes the sequence step-by-step (word by word).
* **Why it matters:** Standard neural networks have no "memory". An LSTM has a hidden state and "gates" (Forget, Input, Output) that allow it to remember early parts of a sentence and apply them to later parts.
* *Example:* In the sentence *"The internet is not working, it is completely terrible,"* the LSTM remembers the negation *"not"* and carries that context across the sequence to understand the overall negative sentiment.

#### 4. Output Layer

* **What it does:** A standard Dense (fully connected) neural network layer with **3 units** (one for each class: `positive`, `negative`, `neutral`).
* **Activation:** We used a `softmax` activation function, which converts the raw output numbers into probabilities that sum up to 1 (e.g., 90% positive, 8% neutral, 2% negative). The class with the highest probability is chosen as the final prediction.

#### 5. Loss Function

* **What it is:** `sparse_categorical_crossentropy`
* **Why it matters:** This function calculates the error between the model's predicted probabilities and the actual true label. "Categorical" is used because we have more than two classes, and "Sparse" is used because our target labels are integers (0, 1, 2) rather than one-hot encoded vectors (like `[0, 1, 0]`). The model's entire goal during training is to minimize this loss.

#### 6. Evaluation Metric

* **What it is:** `accuracy`
* **Why it matters:** It tells us the percentage of times the model's highest probability prediction matched the exact true sentiment label.

---

### Summary: Traditional vs. Sequence Models

Through this pipeline, you have observed the difference between traditional ML and deep learning for text:

* **Traditional (TF-IDF + Logistic Regression):** Great for simple, distinct keywords. It is fast, lightweight, and explainable, but it destroys word order and struggles with sarcasm, negation, and complex grammar.
* **Sequence Models (Embeddings + LSTM):** Preserves word order and learns deep semantic relationships. It requires more compute power and data to train, but it represents the true reality of human language.

==============================================
## Task 6: Attention and Transformer Reflection 
### The Evolution to Modern Generative AI

To wrap up this NLP pipeline, it is important to understand how we evolved from the simple sequence models we just built to the massive language models (like the one you are interacting with right now). Here is the conceptual journey:

**1. Why RNNs struggle with long-term dependencies**
Basic Recurrent Neural Networks (RNNs) process sequences one step at a time, updating a single "hidden state" (memory) at each step. If a sentence is very long, the mathematical calculations required to update this state repeatedly cause the network to suffer from the **"vanishing gradient" problem**. The information from the very first word gets diluted and mathematically shrinks toward zero by the time the model reaches the 50th word.

* *Example:* In the sentence "I grew up in France... [50 words later] ... so I speak fluent ___", a basic RNN will likely have forgotten "France" by the end.

**2. How LSTMs help with memory**
Long Short-Term Memory networks (LSTMs), which we used in Task 5, were invented to solve this exact issue. Instead of just a single hidden state, they introduce a core **Cell State**—think of it like a conveyor belt tha git t runs straight down the entire sequence. They also introduce **Gates** (Forget, Input, and Output gates). These gates act like bouncers, explicitly learning *which* information is important enough to add to the conveyor belt, and *which* irrelevant information should be thrown away, preserving long-term context.

**3. What Attention solves in Sequence-to-Sequence tasks**
Even with LSTMs, early translation models suffered from a "bottleneck." They tried to compress an entire paragraph into one final hidden state vector before translating it.
The **Attention Mechanism** solved this by saying: *"Instead of relying on one final summary, let the model look back at EVERY word in the original sentence at the exact moment it generates a new word."* When translating "The green apple" to French ("La pomme verte"), when the model predicts "verte", Attention allows it to focus heavily on the word "green" rather than looking at the whole sentence equally.

**4. Why Transformers are the foundation of modern Generative AI**
In 2017, Google introduced the Transformer architecture (the "T" in GPT). Transformers threw away the sequential, step-by-step processing of RNNs and LSTMs entirely.

* **Self-Attention:** Instead of reading left-to-right, a Transformer looks at every word in the sentence simultaneously and calculates how each word relates to every other word.
* **Parallelization:** Because it doesn't wait for word 1 to finish before processing word 2, Transformers can be trained in parallel across massive clusters of GPUs.

This ability to process massive amounts of data simultaneously, while perfectly understanding the complex relationships between words regardless of distance, is what unlocked the scale required for modern Generative AI like BERT, GPT, and Gemini.

