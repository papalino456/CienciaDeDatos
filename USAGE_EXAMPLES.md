# Usage Examples

## Using the Trained Model for Inference

Once the pipeline completes, you can use the trained model to encode mechatronics sentences into embeddings.

### Basic Encoding

```python
import torch
from transformers import BertModel
from tokenizers import Tokenizer
import numpy as np
import torch.nn.functional as F

# Load model and tokenizer
model_path = 'artifacts/models/mecha-embed-v1/best/'
tokenizer_path = 'data/tokenizer/tokenizer.json'

model = BertModel.from_pretrained(model_path)
tokenizer = Tokenizer.from_file(tokenizer_path)

# Set to evaluation mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def encode_sentence(text, model, tokenizer, device='cpu'):
    """Encode a single sentence to embedding."""
    # Tokenize
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids]).to(device)
    attention_mask = torch.tensor([[1] * len(encoding.ids)]).to(device)
    
    # Encode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        
        # L2 normalize
        embedding = F.normalize(pooled, p=2, dim=1)
    
    return embedding.cpu().numpy()

# Example usage
sentence = "The PID controller regulates motor speed using feedback from an encoder."
embedding = encode_sentence(sentence, model, tokenizer, device)

print(f"Sentence: {sentence}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")  # Should be ~1.0
```

### Batch Encoding

```python
def encode_batch(sentences, model, tokenizer, device='cpu', max_length=256):
    """Encode multiple sentences efficiently."""
    model.eval()
    
    # Tokenize all
    encodings = [tokenizer.encode(s) for s in sentences]
    
    # Find max length in batch
    max_len = min(max(len(enc.ids) for enc in encodings), max_length)
    
    # Prepare batch
    input_ids = []
    attention_mask = []
    
    for enc in encodings:
        ids = enc.ids[:max_len]
        mask = [1] * len(ids)
        
        # Pad
        pad_len = max_len - len(ids)
        ids = ids + [0] * pad_len
        mask = mask + [0] * pad_len
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
    
    # Encode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        
        # L2 normalize
        embeddings = F.normalize(pooled, p=2, dim=1)
    
    return embeddings.cpu().numpy()

# Example
sentences = [
    "Robot manipulators use inverse kinematics for positioning.",
    "Sensors convert physical quantities into electrical signals.",
    "A PLC controls industrial automation processes.",
    "Stepper motors provide precise positioning control."
]

embeddings = encode_batch(sentences, model, tokenizer, device)
print(f"Encoded {len(sentences)} sentences")
print(f"Embeddings shape: {embeddings.shape}")
```

### Semantic Similarity

```python
def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2.T)  # Already L2 normalized

# Example: Find most similar sentence
query = "How do robots move their arms?"
query_emb = encode_sentence(query, model, tokenizer, device)

corpus = [
    "Robot manipulators use inverse kinematics for positioning.",
    "Sensors measure temperature and pressure.",
    "PLC programming uses ladder logic.",
    "Kinematics describes the motion of robot links."
]

corpus_embs = encode_batch(corpus, model, tokenizer, device)

# Compute similarities
similarities = cosine_similarity(query_emb, corpus_embs)[0]

# Rank by similarity
ranked = sorted(zip(corpus, similarities), key=lambda x: x[1], reverse=True)

print(f"Query: {query}\n")
print("Most similar sentences:")
for i, (sent, sim) in enumerate(ranked, 1):
    print(f"{i}. [{sim:.4f}] {sent}")
```

### Semantic Search

```python
class SemanticSearchEngine:
    """Simple semantic search over a corpus."""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.corpus = []
        self.embeddings = None
    
    def index(self, documents):
        """Index a corpus of documents."""
        self.corpus = documents
        print(f"Encoding {len(documents)} documents...")
        self.embeddings = encode_batch(documents, self.model, self.tokenizer, self.device)
        print("Indexing complete!")
    
    def search(self, query, top_k=5):
        """Search for most similar documents."""
        if self.embeddings is None:
            raise ValueError("No corpus indexed. Call .index() first.")
        
        # Encode query
        query_emb = encode_sentence(query, self.model, self.tokenizer, self.device)
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.corpus[idx],
                'score': float(similarities[idx])
            })
        
        return results

# Example usage
search_engine = SemanticSearchEngine(model, tokenizer, device)

# Index a corpus
corpus = [
    "PID controllers use proportional, integral, and derivative terms.",
    "Encoders provide position feedback for servo motors.",
    "Industrial robots perform repetitive manufacturing tasks.",
    "LiDAR sensors measure distances using laser light.",
    "Hydraulic actuators generate high forces for heavy loads.",
    "SCADA systems monitor and control industrial processes.",
    "Microcontrollers execute embedded firmware in real-time.",
    "Forward kinematics computes end-effector position from joint angles."
]

search_engine.index(corpus)

# Search
results = search_engine.search("How do motors know their position?", top_k=3)

print("Search results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.4f}")
    print(f"   {result['document']}")
```

### Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Encode a corpus
corpus = [
    # Control systems
    "PID controller regulates system output.",
    "Feedback loops improve control accuracy.",
    
    # Sensors
    "Encoders measure rotational position.",
    "LiDAR scans the environment with lasers.",
    
    # Robotics
    "Robot arms have multiple degrees of freedom.",
    "Path planning avoids obstacles.",
    
    # Motors
    "Stepper motors move in discrete steps.",
    "Servo motors use feedback for positioning."
]

embeddings = encode_batch(corpus, model, tokenizer, device)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Print clusters
for cluster_id in range(4):
    print(f"\nCluster {cluster_id}:")
    for i, (sent, label) in enumerate(zip(corpus, clusters)):
        if label == cluster_id:
            print(f"  - {sent}")
```

### Dimensionality Reduction (t-SNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Encode corpus
corpus = [
    "PID feedback control", "Servo motor control",
    "Encoder position sensor", "LiDAR distance sensor",
    "Robot kinematics", "Path planning algorithm",
    "Stepper motor", "Hydraulic actuator"
]

embeddings = encode_batch(corpus, model, tokenizer, device)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

for i, txt in enumerate(corpus):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                 fontsize=9, ha='right')

plt.title('Sentence Embeddings Visualization (t-SNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.tight_layout()
plt.savefig('embeddings_tsne.png', dpi=150)
plt.show()
```

### Document Classification (with fine-tuning)

```python
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SentenceClassifier(nn.Module):
    """Classification head on top of embeddings."""
    
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pool
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        
        # Classify
        logits = self.classifier(pooled)
        return logits

# Example: Fine-tune for topic classification
# (You would need labeled data for this)

num_classes = 8  # 8 mechatronics topics
classifier = SentenceClassifier(model, hidden_size=256, num_classes=num_classes)

# Training loop would go here...
# optimizer = AdamW(classifier.parameters(), lr=3e-5)
# criterion = nn.CrossEntropyLoss()
# ...
```

### Reusable Encoder Class

```python
class MechatronicsEncoder:
    """Easy-to-use wrapper for the trained model."""
    
    def __init__(self, model_path, tokenizer_path, device=None):
        self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, sentences, batch_size=32):
        """Encode sentences to embeddings."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embs = encode_batch(batch, self.model, self.tokenizer, self.device)
            all_embeddings.append(batch_embs)
        
        return np.vstack(all_embeddings)
    
    def similarity(self, sent1, sent2):
        """Compute similarity between two sentences."""
        emb1 = self.encode(sent1)
        emb2 = self.encode(sent2)
        return float(np.dot(emb1, emb2.T)[0, 0])
    
    def most_similar(self, query, candidates, top_k=5):
        """Find most similar sentences from candidates."""
        query_emb = self.encode(query)
        cand_embs = self.encode(candidates)
        
        sims = cosine_similarity(query_emb, cand_embs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        return [(candidates[i], float(sims[i])) for i in top_idx]

# Usage
encoder = MechatronicsEncoder(
    'artifacts/models/mecha-embed-v1/best/',
    'data/tokenizer/tokenizer.json'
)

# Simple encoding
emb = encoder.encode("PID controller for motor speed.")
print(f"Embedding: {emb.shape}")

# Similarity
sim = encoder.similarity(
    "Robot uses inverse kinematics.",
    "Forward kinematics computes position."
)
print(f"Similarity: {sim:.4f}")

# Find similar
results = encoder.most_similar(
    "How do sensors work?",
    ["Encoders measure position.", "Motors generate torque.", "Sensors convert signals."],
    top_k=2
)
for sent, score in results:
    print(f"[{score:.4f}] {sent}")
```

## Tips for Best Results

1. **Input length**: Keep sentences between 10-100 words for best results
2. **Normalization**: Embeddings are L2 normalized; use cosine similarity
3. **Batch size**: Use batch encoding for efficiency with many sentences
4. **Domain**: Model works best on mechatronics-related text
5. **Fine-tuning**: For specific tasks, add a classification head and fine-tune

## Common Use Cases

- **Semantic search**: Find relevant documentation
- **Question answering**: Match questions to answers
- **Duplicate detection**: Find similar technical descriptions
- **Clustering**: Group related concepts
- **Recommendation**: Suggest related articles/topics
- **Auto-tagging**: Classify documents by topic

