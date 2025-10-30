import os
import math
import pickle
from collections import defaultdict, Counter
import re
import random

# ---------------- Helper ----------------

def nested_defaultdict_float():
    return defaultdict(float)

# ---------------- Main Model ----------------

class VectorNaiveBayesTFIDF:
    STOPWORDS = {
        'the','is','am','are','a','an','of','to','in','and','on','for','this',
        'that','with','as','was','were','be','it','by','from','at','or','so',
        'if','but','do','does','did','what','which','who','whom','been','have',
        'has','had','you','your','i','me','my','we','our','us','they','them',
        'their','he','she','his','her','its','about','into','out','up','down',
        'over','under','again','once','then','than','too','very','can','will',
        'just','should','could','would'
    }

    def __init__(self, use_bigrams=True):
        self.vocab = set()
        self.class_totals = defaultdict(float)
        self.class_word_tfidf = defaultdict(nested_defaultdict_float)
        self.doc_counts = defaultdict(int)
        self.total_docs = 0
        self.doc_freq = defaultdict(int)
        self.use_bigrams = use_bigrams

    # -------- Tokenization --------
    def tokenize(self, text):
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        if self.use_bigrams:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            words += bigrams
        return words

    # -------- Vocabulary & Document Frequency --------
    def build_vocab_and_df(self, dataset):
        for text, _ in dataset:
            words = set(self.tokenize(text))
            self.vocab.update(words)
            for w in words:
                self.doc_freq[w] += 1

    # -------- TF-IDF Computation --------
    def compute_tfidf(self, vec, total_docs):
        tfidf_vec = defaultdict(float)
        length = sum(vec.values()) or 1
        for word, count in vec.items():
            df = self.doc_freq.get(word, 1)
            idf = math.log((total_docs + 1) / (df + 1)) + 1
            tfidf_vec[word] = (count / length) * idf
        return tfidf_vec

    def vectorize(self, text):
        vec = defaultdict(int)
        for w in self.tokenize(text):
            if w in self.vocab:
                vec[w] += 1
        return self.compute_tfidf(vec, self.total_docs)

    # -------- Training --------
    def train(self, dataset):
        self.build_vocab_and_df(dataset)
        self.total_docs = len(dataset)
        for text, label in dataset:
            self.doc_counts[label] += 1
            tfidf_vec = self.vectorize(text)
            for word, val in tfidf_vec.items():
                self.class_word_tfidf[label][word] += val
                self.class_totals[label] += val

    # -------- Prediction --------
    def predict(self, text):
        tfidf_vec = self.vectorize(text)
        scores = {}
        vocab_size = len(self.vocab)
        for label in self.class_word_tfidf:
            log_prob = math.log(self.doc_counts[label] / self.total_docs)
            total = self.class_totals[label] + vocab_size
            for word, val in tfidf_vec.items():
                word_val = self.class_word_tfidf[label].get(word, 0) + 1
                log_prob += val * math.log(word_val / total)
            scores[label] = log_prob
        return max(scores, key=scores.get)

    # -------- Save & Load --------
    def save(self, filename):
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        if not filename.startswith('models/'):
            filename = os.path.join(models_dir, os.path.basename(filename))
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        if not filename.startswith('models/'):
            filename = os.path.join('models', os.path.basename(filename))
        with open(filename, 'rb') as f:
            return pickle.load(f)

# ---------------- Dataset Parsers ----------------

def parse_sentiment_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                text, label = line.strip().split('|', 1)
                data.append((text.strip(), label.strip().lower()))
    return data

def parse_priority_file(filename):
    data = []
    local_vars = {}
    with open(filename, 'r', encoding='utf-8') as f:
        code = f.read()
        exec(code, {}, local_vars)
    if 'training_data' in local_vars:
        for text, label in local_vars['training_data']:
            data.append((text.strip(), label.strip().lower()))
    else:
        raise ValueError("training_data not found in file")
    return data

# ---------------- Model Loader ----------------

def vector_tfidf_model(model_file, dataset_files, parser):
    if not model_file.startswith('models/'):
        model_file = os.path.join('models', os.path.basename(model_file))
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        model_mtime = os.path.getmtime(model_file)
        dataset_updated = any(os.path.getmtime(f) > model_mtime for f in dataset_files)
        if not dataset_updated:
            try:
                print(f"Loading pre-existing model from {model_file}...")
                return VectorNaiveBayesTFIDF.load(model_file)
            except EOFError:
                print(f"Warning: {model_file} is empty/corrupt, retraining...")

    data = []
    for file in dataset_files:
        data.extend(parser(file))

    print(f"Training new model and saving to {model_file}...")
    model = VectorNaiveBayesTFIDF()
    model.train(data)
    model.save(model_file)
    return model

# ---------------- Evaluation ----------------

def evaluate_model_manual(model, test_data):
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    for text, true_label in test_data:
        predicted_label = model.predict(text)
        total += 1
        class_total[true_label] = class_total.get(true_label, 0) + 1
        if predicted_label == true_label:
            correct += 1
            class_correct[true_label] = class_correct.get(true_label, 0) + 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    print("\nPer-class accuracy:")
    for label in class_total:
        acc = (class_correct.get(label, 0) / class_total[label]) * 100
        print(f"  {label}: {acc:.2f}%")

    return accuracy

# ---------------- Utility: Check Data Balance ----------------

def print_data_balance(dataset):
    counter = Counter(label for _, label in dataset)
    print("\nDataset balance:")
    for label, count in counter.items():
        print(f"  {label}: {count} samples")

def evaluate_model(model, test_data):
    correct = 0
    total = len(test_data)
    print("\n--- Test Results ---\n")
    for text, true_label in test_data:
        pred = model.predict(text)
        print(f"Text: {text}")
        print(f"True: {true_label}, Predicted: {pred}\n")
        if pred == true_label:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Overall Accuracy: {accuracy*100:.2f}%")




# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # dataset = parse_sentiment_file('train_sentiment.txt')
    dataset = parse_priority_file('TrainPriority.txt')
    print("Total samples:", len(dataset))
    random.shuffle(dataset)
    split_index = int(0.8 * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    model = VectorNaiveBayesTFIDF()
    model.train(train_data)

    # evaluate_model_manual(model, test_data)

    # print("\nExample prediction:")
    # example_text = "The teaching is average but could be improved with more practical examples."
    # print(f"Text: {example_text}")
    # print(f"Predicted: {model.predict(example_text)}")

    evaluate_model(model, test_data)
