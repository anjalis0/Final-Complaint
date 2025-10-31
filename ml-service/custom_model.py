import os
import math
import pickle
from collections import defaultdict, Counter
import re
import random
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        logging.info("Initializing VectorNaiveBayesTFIDF model")
        self.vocab = set()
        self.class_totals = defaultdict(float)
        self.class_word_tfidf = defaultdict(nested_defaultdict_float)
        self.doc_counts = defaultdict(int)
        self.total_docs = 0
        self.doc_freq = defaultdict(int)
        self.use_bigrams = use_bigrams
        logging.info(f"Model initialized with bigrams enabled: {use_bigrams}")
        logging.info(f"Loaded {len(self.STOPWORDS)} stopwords")

    # -------- Tokenization --------
    def tokenize(self, text):
        logging.debug(f"Tokenizing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        original_word_count = len(words)
        words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        filtered_word_count = len(words)
        
        if self.use_bigrams:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            words += bigrams
            logging.debug(f"Generated {len(bigrams)} bigrams")
        
        logging.debug(f"Tokenization complete: {original_word_count} -> {filtered_word_count} words (after filtering), final features: {len(words)}")
        return words

    # -------- Vocabulary & Document Frequency --------
    def build_vocab_and_df(self, dataset):
        logging.info(f"Building vocabulary and document frequencies from {len(dataset)} documents")
        for i, (text, _) in enumerate(dataset):
            # if i % 1000 == 0 and i > 0:
            #     logging.info(f"Processed {i}/{len(dataset)} documents for vocabulary building")
            words = set(self.tokenize(text))
            self.vocab.update(words)
            for w in words:
                self.doc_freq[w] += 1
        
        logging.info(f"Vocabulary built: {len(self.vocab)} unique features")
        logging.info(f"Top 10 most frequent features: {sorted(self.doc_freq.items(), key=lambda x: x[1], reverse=True)[:10]}")

    # -------- TF-IDF Computation --------
    def compute_tfidf(self, vec, total_docs):
        logging.debug(f"Computing TF-IDF for vector with {len(vec)} unique features")
        tfidf_vec = defaultdict(float)
        length = sum(vec.values()) or 1
        for word, count in vec.items():
            df = self.doc_freq.get(word, 1)
            idf = math.log((total_docs + 1) / (df + 1)) + 1
            tfidf_vec[word] = (count / length) * idf
        logging.debug(f"TF-IDF computation complete: {len(tfidf_vec)} features with non-zero scores")
        return tfidf_vec

    def vectorize(self, text):
        logging.debug("Starting text vectorization")
        vec = defaultdict(int)
        tokens = self.tokenize(text)
        for w in tokens:
            if w in self.vocab:
                vec[w] += 1
        logging.debug(f"Raw vector created: {len(vec)} features from {len(tokens)} tokens")
        tfidf_result = self.compute_tfidf(vec, self.total_docs)
        logging.debug(f"Vectorization complete")
        return tfidf_result

    # -------- Training --------
    def train(self, dataset):
        logging.info("=" * 50)
        logging.info("STARTING MODEL TRAINING")
        logging.info("=" * 50)
        
        logging.info("Step 1: Building vocabulary and document frequencies")
        self.build_vocab_and_df(dataset)
        self.total_docs = len(dataset)
        
        logging.info("Step 2: Processing training documents and computing TF-IDF scores")
        for i, (text, label) in enumerate(dataset):
            # if i % 1000 == 0:
            #     logging.info(f"Processing document {i+1}/{len(dataset)} (Label: {label})")

            self.doc_counts[label] += 1
            tfidf_vec = self.vectorize(text)
            for word, val in tfidf_vec.items():
                self.class_word_tfidf[label][word] += val
                self.class_totals[label] += val
        
        logging.info("Training complete!")
        logging.info(f"Total documents processed: {self.total_docs}")
        logging.info(f"Classes found: {list(self.doc_counts.keys())}")
        for label, count in self.doc_counts.items():
            logging.info(f"  - {label}: {count} documents ({count/self.total_docs*100:.1f}%)")
        logging.info(f"Vocabulary size: {len(self.vocab)}")
        logging.info("=" * 50)

    # -------- Prediction --------
    def predict(self, text):
        logging.info(f"Making prediction for text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        tfidf_vec = self.vectorize(text)
        logging.info(f"Text vectorized into {len(tfidf_vec)} features")
        
        scores = {}
        vocab_size = len(self.vocab)
        
        for label in self.class_word_tfidf:
            logging.debug(f"Calculating probability for class: {label}")
            
            # Prior probability
            prior = self.doc_counts[label] / self.total_docs
            log_prob = math.log(prior)
            logging.debug(f"  Prior probability: {prior:.4f} (log: {log_prob:.4f})")
            
            # Likelihood calculation
            total = self.class_totals[label] + vocab_size
            for word, val in tfidf_vec.items():
                word_val = self.class_word_tfidf[label].get(word, 0) + 1
                likelihood = word_val / total
                log_prob += val * math.log(likelihood)
            
            scores[label] = log_prob
            logging.debug(f"  Final log probability for {label}: {log_prob:.4f}")
        
        predicted_class = max(scores, key=scores.get)
        logging.info(f"Prediction scores: {dict(scores)}")
        logging.info(f"PREDICTED CLASS: {predicted_class}")
        return predicted_class

    # -------- Save & Load --------
    def save(self, filename):
        logging.info(f"Saving model to: {filename}")
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        if not filename.startswith('models/'):
            filename = os.path.join(models_dir, os.path.basename(filename))
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Model successfully saved to: {filename}")

    def load(filename):
        logging.info(f"Loading model from: {filename}")
        if not filename.startswith('models/'):
            filename = os.path.join('models', os.path.basename(filename))
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model successfully loaded from: {filename}")
        logging.info(f"Loaded model stats: {len(model.vocab)} vocab, {len(model.class_word_tfidf)} classes, {model.total_docs} training docs")
        return model

# ---------------- Dataset Parsers ----------------

def parse_sentiment_file(filename):
    logging.info(f"Parsing sentiment file: {filename}")
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if '|' in line:
                text, label = line.strip().split('|', 1)
                data.append((text.strip(), label.strip().lower()))
                if line_num % 50 == 0:
                    logging.info(f"Parsed {line_num} lines from sentiment file")
    logging.info(f"Sentiment file parsing complete: {len(data)} samples loaded")
    return data

def parse_priority_file(filename):
    logging.info(f"Parsing priority file: {filename}")
    data = []
    local_vars = {}
    with open(filename, 'r', encoding='utf-8') as f:
        code = f.read()
        logging.info("Executing Python code from priority file")
        exec(code, {}, local_vars)
    if 'training_data' in local_vars:
        line_num=1
        for text, label in local_vars['training_data']:
            if line_num % 500 == 0:
                logging.info(f"Parsed {line_num} lines from priority file")
            data.append((text.strip(), label.strip().lower()))
            line_num += 1
        logging.info(f"Priority file parsing complete: {len(data)} samples loaded")
    else:
        logging.error("training_data variable not found in priority file")
        raise ValueError("training_data not found in file")
    return data

# ---------------- Model Loader ----------------

def vector_tfidf_model(model_file, dataset_files, parser):
    logging.info("=" * 60)
    logging.info("VECTOR TF-IDF MODEL LOADER")
    logging.info("=" * 60)
    
    if not model_file.startswith('models/'):
        model_file = os.path.join('models', os.path.basename(model_file))
    
    logging.info(f"Checking for existing model: {model_file}")
    
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        logging.info("Existing model found, checking if datasets have been updated")
        model_mtime = os.path.getmtime(model_file)
        dataset_updated = any(os.path.getmtime(f) > model_mtime for f in dataset_files)
        
        if not dataset_updated:
            try:
                logging.info(f"Dataset not updated, loading pre-existing model from {model_file}")
                return VectorNaiveBayesTFIDF.load(model_file)
            except EOFError:
                logging.warning(f"Model file {model_file} is empty/corrupt, will retrain")
        else:
            logging.info("Dataset has been updated since model was saved, will retrain")
    else:
        logging.info("No existing model found or model file is empty")

    logging.info("Loading and parsing dataset files")
    data = []
    for file in dataset_files:
        logging.info(f"Parsing dataset file: {file}")
        file_data = parser(file)
        data.extend(file_data)
        logging.info(f"Added {len(file_data)} samples from {file}")

    logging.info(f"Total dataset size: {len(data)} samples")
    logging.info(f"Training new model and saving to {model_file}")
    
    model = VectorNaiveBayesTFIDF()
    model.train(data)
    model.save(model_file)
    
    logging.info("Model loading/training complete!")
    return model

# ---------------- Evaluation ----------------

def evaluate_model_manual(model, test_data):
    logging.info("=" * 50)
    logging.info("STARTING MODEL EVALUATION")
    logging.info("=" * 50)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    for i, (text, true_label) in enumerate(test_data):
        if i % 10 == 0:
            logging.info(f"Evaluating sample {i+1}/{len(test_data)}")
        
        predicted_label = model.predict(text)
        total += 1
        class_total[true_label] = class_total.get(true_label, 0) + 1
        if predicted_label == true_label:
            correct += 1
            class_correct[true_label] = class_correct.get(true_label, 0) + 1

    accuracy = correct / total if total > 0 else 0
    logging.info(f"Evaluation complete!")
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    logging.info("Per-class accuracy:")
    for label in class_total:
        acc = (class_correct.get(label, 0) / class_total[label]) * 100
        logging.info(f"  {label}: {acc:.2f}% ({class_correct.get(label, 0)}/{class_total[label]})")

    return accuracy

# ---------------- Utility: Check Data Balance ----------------

def print_data_balance(dataset):
    logging.info("Analyzing dataset balance:")
    counter = Counter(label for _, label in dataset)
    logging.info("Dataset balance:")
    for label, count in counter.items():
        percentage = (count / len(dataset)) * 100
        logging.info(f"  {label}: {count} samples ({percentage:.1f}%)")

def evaluate_model(model, test_data):
    logging.info("=" * 50)
    logging.info("DETAILED MODEL EVALUATION")
    logging.info("=" * 50)
    
    correct = 0
    total = len(test_data)
    logging.info(f"Evaluating model on {total} test samples")
    # Print detailed info for every test sample (no filtering)
    for i, (text, true_label) in enumerate(test_data):
        logging.info(f"\n--- Test Sample {i+1}/{total} ---")
        pred = model.predict(text)
        logging.info(f"Text: {text}")
        logging.info(f"True: {true_label}, Predicted: {pred}")
        if pred == true_label:
            correct += 1
            logging.info("✓ CORRECT")
        else:
            logging.info("✗ INCORRECT")
    
    accuracy = correct / total if total > 0 else 0
    logging.info(f"\n" + "=" * 50)
    logging.info(f"FINAL EVALUATION RESULTS")
    logging.info(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    logging.info("=" * 50)




# ---------------- Example Usage ----------------
if __name__ == "__main__":
    logging.info("🚀 Starting VectorNaiveBayesTFIDF Example")
    logging.info("=" * 60)

    def run_dataset_pipeline(dataset_file, parser, model_name, train_fraction=0.99):
        """Parse, train, evaluate and save a model for a single dataset file.

        dataset_file: path to dataset file
        parser: function to parse the dataset file and return list[(text, label)]
        model_name: base name used to save the trained model under `models/`
        """
        logging.info("\n" + "-" * 60)
        logging.info(f"Pipeline start for dataset: {dataset_file}")
        try:
            data = parser(dataset_file)
        except FileNotFoundError:
            logging.error(f"Dataset file not found: {dataset_file}. Skipping.")
            return None
        except Exception as e:
            logging.exception(f"Failed to parse dataset {dataset_file}: {e}")
            return None

        logging.info(f"Total samples loaded from {dataset_file}: {len(data)}")
        if len(data) == 0:
            logging.warning(f"No samples found in {dataset_file}. Skipping training.")
            return None

        print_data_balance(data)

        random.shuffle(data)
        # Validate train_fraction
        try:
            train_fraction = float(train_fraction)
        except Exception:
            logging.warning(f"Invalid train_fraction provided ({train_fraction}), defaulting to 0.99")
            train_fraction = 0.99

        if train_fraction <= 0 or train_fraction >= 1:
            logging.warning(f"train_fraction should be between 0 and 1 (exclusive). Received {train_fraction}. Clamping to valid range.")
            train_fraction = min(max(train_fraction, 0.01), 0.99)

        split_index = int(train_fraction * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        logging.info(f"Training set: {len(train_data)} samples")
        logging.info(f"Test set: {len(test_data)} samples")

        logging.info("Creating and training model...")
        model = VectorNaiveBayesTFIDF()
        try:
            model.train(train_data)
        except Exception:
            logging.exception("Error during training. Aborting this dataset pipeline.")
            return None

        if len(test_data) > 0:
            logging.info("Starting model evaluation...")
            try:
                evaluate_model(model, test_data)
            except Exception:
                logging.exception("Error during evaluation (continuing).")

        # Save the model
        model_filename = f"{model_name}.pkl" if model_name.endswith('.pkl') else f"{model_name}.pkl"
        try:
            model.save(os.path.join('models', model_filename))
        except Exception:
            logging.exception(f"Failed to save model to models/{model_filename}")

        logging.info(f"Pipeline complete for dataset: {dataset_file}")
        logging.info("-" * 60 + "\n")
        return model

    # Run pipelines for both sentiment and priority datasets
    # Note: parse_priority_file expects the file to define a `training_data` variable.
    todo_runs = [
        ("TrainSentiment.txt", parse_sentiment_file, "model_sentiment"),
        ("TrainPriority.txt", parse_priority_file, "model_priority"),
    ]

    # Parse CLI arguments for dynamic train/test split
    parser_arg = argparse.ArgumentParser(description='Train TF-IDF Naive Bayes models with dynamic train/test split')
    parser_arg.add_argument('--train-fraction', '-t', type=float, default=0.99,
                            help='Fraction of data to use for training (0 < fraction < 1). Default: 0.99')
    args = parser_arg.parse_args()

    train_fraction = args.train_fraction
    logging.info(f"Using train fraction: {train_fraction}")

    for file, parser, model_name in todo_runs:
        run_dataset_pipeline(file, parser, model_name, train_fraction=train_fraction)

    logging.info("🎉 All pipelines complete!")
    logging.info("=" * 60)
