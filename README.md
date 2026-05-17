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
