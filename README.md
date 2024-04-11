# Detecting_Your_MBTI_Using_NLP

## PersonaLens: MBTI Analysis with Transformer Technology
The members of our group are Xuetong Tang(xtang34), Yicheng Lu(ylu204), Ke Zhang(kzhan176), and Dongyan Sun (dsun35).
### Introduction
Our project aims to revolutionize the way Myers-Briggs Type Indicator (MBTI) assessments are conducted. Traditional MBTI assessments, with their lengthy questionnaires, can be time-consuming and daunting for users.

By training the model on our dataset containing various users’ post sentences along with their corresponding MBTI types, the aim is to create a system that can accurately classify posts text inputs into one of the 16 MBTI personality types. Our solution would leverage advanced Transformer-based natural language processing (NLP) technology to analyze and interpret users' textual responses swiftly and accurately. By integrating this cutting-edge AI approach, we propose a streamlined, efficient, and user-friendly platform for personality assessment. Nowadays, MBTI personality test definitely is a trending topic. Motivated by the growing interest in understanding how language reflects personality traits and the potential applications in various fields such as social media analysis, marketing, and personalized recommendation systems, this project aims to explore the potential application of MBTI by analyzing user’s MBTI based on their social media posts. Thus, the decision to undertake this project was made. This project is a multi-classification problem. Here, our target variable includes 16 classes each representing one specific MBTI personality. Later on, in our preprocessing and modeling part, we would also consider transforming our problem into 4 binary classification problems where we would perform binary classification on each key dichotomies of MBTI personality code.
### Related Works
URL for the paper: [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9183090)

The paper "Analysis of Personality Traits using Natural Language Processing and Deep Learning" by Tejas Pradhan et al. is a related work which discusses automating personality assessments using neural networks and NLP, focusing on the MBTI framework. Traditional personality tests, often lengthy and costly, are prone to manipulation. The authors propose a more efficient and less biased method using social media data and image-based user responses. The process involves text mining for data cleaning and feature extraction, followed by the application of machine learning and deep learning models for personality classification. Specifically, they train models like Support Vector Machines, Naive Bayes, Random Forests, and Convolutional Neural Networks (CNN) to predict personality types, achieving an accuracy of up to 81.4% with CNN. The study also incorporates the development of an interactive website for administering the test and collecting user feedback to refine the model further.
### Data
Data link: [link](https://www.kaggle.com/datasets/datasnaek/mbti-type)

Data description: Our dataset is collected from Kaggle. It was originally collected through the PersonalityCafe forum. This website  provides a large selection of people and their MBTI personality type, as well as corpses of their posts. This dataset contains over 8600 rows of data. Each row consist of the the person’s 4 letter MBTI and the most recent 50 things they have posted (Each entry separated by "|||").

Preprocessing:

1. NLP preprocessing methods: 

(a). TF-IDF(Term Frequency-Inverse Document Frequency) in MLP base line model:

TF (Term Frequency) represents the frequency of a word in a document. This is calculated as the number of times a word appears in a document divided by the total number of words in that document. It provides a measure of how often the term appears in the document. On the other hand, IDF (Inverse Document Frequency) is calculated as the logarithm of the number of documents in the corpus divided by the number of documents that contain the word. IDF decreases as the number of documents containing the word increases, which helps to adjust for the fact that some words appear more frequently in general. The overall TF-IDF score of a word in a document is the product of its TF and IDF scores. High TF-IDF scores suggest a term is more relevant within the given document, while lower scores indicate lesser relevance.
By using the TF-IDF method, In the training stage, we would create a dictionary which includes all the words in our corpus. Then, for each word, we would vectorize it by calculating its TF-IDF score.  

(b). Word2Vec for LSTM and Transformer 

Word2Vec: Developed by researchers at Google, Word2Vec models can capture complex contextual word relationships in natural language processing (NLP) applications. Word2Vec uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. Word2Vec creates a dense vector for each word, typically of several hundred dimensions. These vectors aim to place semantically similar words close to one another in the vector space.

By using Word2Vec method, after fitting it on our training dataset, each word in the vocabulary is associated with a dense vector representation (word embedding) in the vector space. Words with similar meanings or that often appear in similar contexts will have embeddings that are close together in the vector space. Such that, we will get a  resulting word embedding  matrix for each corpus (each row of our dataset) where each row corresponds to the vector representation of a word in the vocabulary. The dimensionality of these vectors would depend on the chosen size of the embedding space, which is typically a parameter set before training the Word2Vec model.

2. Other preprocessing method: 

Sample imbalance: Up sampling and downsampling: Since the distribution of 16 MBTI personality types is unbalanced, we would do up or down sampling based on our EDA result.

Stratified K-fold: We are also considering using Stratified K-fold validation in modeling tuning stage to side step the sample imbalance issue.

Create separate labels of MBTI personality type: We would create 4 new binary class labels representing each letter of MBTI separately: E/I, N/S, F/T , P/J.
### Methodology
We plan to implement a transformer-based approach to classify the MBTI personality type of individuals based on their forum posts. We will integrate a classification layer designed to address the four distinct components of MBTI, each targeting one of the key dichotomies of MBTI: Extraversion/Introversion (E/I), Sensing/Intuition (S/N), Thinking/Feeling (F/T), and Judging/Perceiving (J/P). This structured approach allows for nuanced analysis and classification of each personality aspect, leveraging the transformer's ability to extract relevant features from varied text inputs.

This method is chosen because transformers perform well in capturing long-range dependencies in text. Given that the lengths of posts can vary significantly—from very short to extremely long—transformers are particularly well-suited for extracting the essential context necessary for accurate classification predictions. Additionally, the self-attention mechanism within transformers allows them to evaluate the importance of each word in relation to others within the post. This capability is crucial for comprehending the overall sentiment, thematic elements, and subtle expressions in the posts, all of which are vital for providing deep insights into an individual’s MBTI type. 

If the proposed transformer approach encounters issues, we can consider the following alternatives that potentially maintain a relatively high effectiveness of performing the classification task:

1. LSTM: This approach allows the handling of the varied lengths of forum posts. To enhance performance, we could stack multiple layers or implement a bidirectional approach, which might capture more complex patterns in the data.

2. Combining CNN and LSTM: Although our referenced work uses CNNs for classification, this approach alone is unable to robustly handle the varied-length inputs. We propose that by combining CNNs and LSTMs, the model could effectively capture both local and global contexts. 

3. Adding a self-attention mechanism to the alternative approaches can potentially boost their performance. The self-attention mechanism, enabling the model to assess the importance of each word in relation to others within the post, compensates for their potential shortcomings in understanding overall context.

### Metrics
Our project's success hinges on a comprehensive evaluation strategy, focusing on the model’s precision in forecasting MBTI types. To thoroughly assess our model, we will employ an array of metrics: accuracy, F1-score (macro, micro, and weighted), precision, recall, log loss, and Jaccard score. Each of these metrics provides a unique lens through which to view the model's performance, offering a nuanced understanding of its predictive capabilities and areas for improvement.

For experiment, we aim to reserve 10% of our data set, approximately 90 data points, exclusively for testing. We will feed these unseen data to our model and compute the aforementioned metrics and gain insightful feedback on its real-world applicability. And In benchmarking our progress, we will compare our model's performance against established approaches. The baseline for our comparison will be a multi-level perceptron (MLP) equipped with a softmax layer, a standard for classification tasks. Additionally, we will consider the convolutional neural network (CNN) method proposed by Pradhan et al. as an advanced benchmark. These comparisons will help in contextualizing our model's efficiency and effectiveness in the realm of personality prediction.

Setting clear performance targets, we anticipate our baseline model to achieve around 60% accuracy, reflecting initial capabilities. However, our project aspires to surpass this, setting a target accuracy of 80%, with an ambitious stretch goal of reaching 90%. These goals are not just numerical benchmarks but represent significant improvements in the predictive accuracy and reliability of our MBTI classification model, ensuring a robust, user-friendly, and insightful personality assessment tool.

### Ethics

Why is Deep Learning a good approach to this problem?

Deep learning is an effective approach for classifying MBTI types, as the task we are facing is based on the content of forum posts. This method works well due to its ability to learn from complex and varied aspects of language used by individuals. Deep learning models, capable of extracting both local and global features, and contextual patterns, can dynamically learn the details of how different personality types manifest in text. These models can potentially extract small linguistic details that strongly relate to specific MBTI types, making deep learning an ideal choice for this classification task.

What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?

Our dataset is sourced from posts on an MBTI-focused forum. This origin may naturally lead to over-representation of individuals already knowledgeable about MBTI, skewing both demographic and psychographic profiles. Additionally, certain MBTI types may be more inclined to participate in forum discussions, contributing to selection bias and potentially resulting in an imbalanced dataset. The participants’ prior understanding of MBTI concepts might also influence their language use, as individuals could unconsciously mimic the stereotypical ways of expressing their MBTI identity. This behavior could reinforce biases and lead the algorithm to overfit these stereotypes, hindering its ability to generalize to texts from environments other than MBTI forums.

### Division of labor
Ke Zhang: Related Work, Metrics
Yicheng Lu: Methodology, Ethics
Xuetong Tang: Data, Preprocessing
Dongyan Sun: Introduction


