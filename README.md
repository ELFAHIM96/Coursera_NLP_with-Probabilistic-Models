# Natural Language Processing with Probabilistic Models
**********************************
### Week 1 [Autocorrect and Minimum Edit distance](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201)
*****************************
<p style="padding-left: 30px;">In this week I learned how identify misspelled words, and what is an additive distance, I learned how to build a model to perform an autocorrect and how to calculate probabilities of the correct word.</p>
<p><br /><strong>autocorrect</strong>: is an application that changes the misspelled words to correct ones</p>
<p><strong>minimum Edit Distance</strong> <span style="color: #ff0000;">(Levenshtein distance)</span>: the lowest number of operations needed to tranform one strings into the <br />other. it could be sused in many application like spelling correction, document similatity and machine translation</p>
<p><strong>dynamic programming:</strong> You first solve the smallest subproblem first and then reusing that result you solve the next biggest subproblem<br />then I buil dmy own spellcheker to correct misspelled words.</p>

- [1.Week 1](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/tree/main/NLP-with-Classification-Vector-Spaces/C1_week1)
  - [1.1 Resume Courses](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Cours)
      - [Autocorrect](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Cours)
      - [Building the model](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Cours)
      - [Building the model II](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Cours)
      - [Minimum edit distance](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Cours)
    ### Python Code
  - [1.2 lecture Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/Code)
  - [1.3 Assignment Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%201/assignment)
**************************************
### Week 2 [Part of Speech Tagging and Hidden Markov Model (HMM)](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202)
***************************************
<p>In this Week I learned about Markov chains and hidden Markov models then<br />I use HMM to create part of speech tags for a wall street jornal text corpera</p>
<ul>
<li><strong>Part of speech Tagging</strong></li>
<li><strong>Markov chains models</strong></li>
<li><strong>Hidden Markov models</strong></li>
<li><strong>Viterbi algorithm</strong></li>
<li><strong>Exemple</strong></li>
<li><strong>Codding assignment!</strong></li>
</ul>
<p>Part of speech refers to the category of words of the lexical terms in a language<br />for <strong>exemple (verb, adjective, adverb, pronoun, preposition...)</strong> but wring <br />this terms can become cumbersome during text analysis. so we are going to use a short <br />representation called tags to represent this categories. the process of assigning this tags to the words<br />of a sentence or corpus is referred to as parts of speech tagging (POS)tagging for shorts.<br />Markov chains: they are type of stochastic model that describes a sequences of possibles evnets<br />to get the probabilit&eacute; of each event it needs only the states of the previous events<br /> <strong>transition probabilities</strong>: tell about the chances of going from state (POS tag) to another (POS tag)<br />to calculate this probailities:</p>
<p>1. count occurrences of tag pairs</p>
<p>2 calculate the probabilities using the counts =nbr of occurences devides by all the occurences</p>
<p><strong>the markov property</strong>: states that the probability of the next event only depends<br />on the current event.<br /><strong>Hidden Markov Models: (HMM)</strong> implies that states are hidden or not directly observable from data.</p>
<p><br /><strong>Emission Probabilies:</strong> it's an addition probabilies in hidden satetes, and refers to the transition from <br />the hidden states of your hidden Markov model (verb, Noun, O ..) to the observables ro the words <br />of the corpus (hello how are doing) Table = row for the hidden states =column is designated for <br />each of the observables.</p>
<p><br/><strong>The Viterbi algorithm:</strong> is graph algorithm picturing the problem we want to solve on the graph(initialization step, forward pass, Backward pass)</p>

- [1.Week 2]()
  - [1.1 Resume Courses](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
      - [Part of Speech Tagging](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
      - [Markov Chains ](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
      - [Hidden Markov Models](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
      - [Markov Chains and POS Tags](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
      - [The Viterbi Algorithm](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Cours)
### Python Code
  - [1.2 lecture Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Code)
  - [1.3 Assignment Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%202/Code)
 *********************************************************
### Week 3 [Autocomplete and Language Models](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203)
*********************************************************
In ths week we learned the following concept:
<ul>
<li>
<p>N-Grams and probabilities</p>
</li>
<li>
<p>Approximate sentence probability from N-Grams</p>
</li>
<li>
<p>Build a language model from a corpus</p>
</li>
<li>
<p>Fix missing information</p>
</li>
<li>
<p>Out of vocabulary words with &lt;UNK&gt;</p>
</li>
<li>
<p>Missing N-Gram in corpus with smoothing, backoff and interpolation</p>
</li>
<li>
<p>Evaluate language model with perplexity</p>
</li>
<li>
<p>Coding assignment!</p>
</li>
</ul>

- [1.Week 3](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/tree/main/NLP-with-Classification-Vector-Spaces/C1_week3)
  - [1.1 Resume Courses](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Cours)
      - [N-Grams Overview](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Cours)
      - [N-grams and Probabilities](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Cours)
      - [The N-gram Language Model](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Cours)
      - [Out of Vocabulary Words](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Cours)
### Python Code
  - [1.2 lecture Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/Code)
  - [1.3 Assignment Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%203/assignment)
 
 ### Week 4 [word embedding with neural networks](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%204)
<p>In this week I learned about how word embedding carry the semantic meaningof words, which make them much more powerful <br />the one hot encoding and one hot vector<br /><strong>One hot encoding</strong>: is a type of vector representation in which all of the elements in a vector are 0, except for one has 1.<br /><strong>One hot vector</strong> : is a 1 &times; N matrix (vector) used to distinguish each word in a vocabulary from every other word in the vocabulary<br />word embedding with NN: <br /><strong>Word Embedding Methods</strong><br /><span style="background-color: #ffffff;">Classical Methods</span><br /><strong>word2vec</strong> (Google, 2013)<br /><strong><span style="color: #ff0000;">Continuous bag-of-words (CBOW):</span></strong> the model learns to predict the center word given some context words.<br /><strong><span style="color: #ff00ff;">Continuous skip-gram / Skip-gram with negative sampling (SGNS):</span></strong> the model learns to predict the words surrounding a given input word.<br /><strong><span style="color: #ff0000;">Global Vectors (GloVe) (Stanford, 2014):</span></strong> factorizes the logarithm of the corpus's word co-occurrence matrix, similar to the count matrix you&rsquo;ve used before.</p>
<p><strong><span style="color: #800080;">fastText (Facebook, 2016)</span></strong>: based on the skip-gram model and takes into account the structure of words by representing words as an n-gram of characters. It supports out-of-vocabulary (OOV) words.</p>
<p>Deep learning, contextual embeddings<br /> In these more advanced models, words have different embeddings depending on their context. You can download <strong>pre-trained embeddings</strong> for the following models.</p>
<ul>
<li><strong>BERT (Google, 2018):</strong></li>
<li><strong>ELMo (Allen Institute for AI, 2018)</strong></li>
<li><strong>GPT-2 (OpenAI, 2018)</strong></li>
</ul>

- [1.Week 4](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%204)
  - [1.1 Resume Courses](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%204/Cours)
      - [Word Embeddings ](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/blob/main/NLP-with-Classification-Vector-Spaces/C1_week4/resume_cours%20W4/K-nearest%20neighbors%20_%20Coursera.pdf)
      - [Word Embedding Methods](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/blob/main/NLP-with-Classification-Vector-Spaces/C1_week4/resume_cours%20W4/Hash%20tables%20and%20hash%20functions%20_%20Coursera.pdf)
      - [Architecture for the CBOW Model](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/blob/main/NLP-with-Classification-Vector-Spaces/C1_week4/resume_cours%20W4/Transforming%20word%20vectors%20_%20Coursera.pdf)
      - [How to Create Word Embeddings](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/blob/main/NLP-with-Classification-Vector-Spaces/C1_week4/resume_cours%20W4/Searching%20documents%20_%20Coursera.pdf)
      - [Extracting Word Embedding Vectors](https://github.com/ELFAHIM96/Cousera-NLP-with-Classification-Vector-Spaces/blob/main/NLP-with-Classification-Vector-Spaces/C1_week4/resume_cours%20W4/Multiple%20Planes%20_%20Coursera.pdf)
### Python Code
  - [1.2 lecture Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%204/Code)
  - [1.3 Assignment Code](https://github.com/ELFAHIM96/Coursera_NLP_with-Probabilistic-Models/tree/main/NLP_with%20Probabilistic%20Models/Week%204/assignment)
### Certificate 
![image](https://user-images.githubusercontent.com/65721811/208101290-04cae142-4de4-40c3-b475-c535fc44a1ef.png)


