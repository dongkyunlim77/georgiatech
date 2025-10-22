# Recurrent Neural Networks

## Types of Recurrent Neural Networks

We can divide the applications of RNNs into four main categories:

### One-to-One
- Takes a single input and produces a single output
- Like feedforward neural networks
- Example: Image classification (1 image → 1 label)

### Many-to-One
- Takes a sequence as input but produces a single output
- Example: Sentiment analysis (many words → 1 sentiment)
- Used for character prediction tasks

### One-to-Many
- Takes a single input and produces a sequence
- Example: Image captioning (1 image → sequence of words)

### Many-to-Many
- Takes a sequence and produces a sequence
- Examples:
  - Translation (English sentence → French sentence)
  - Video frame prediction (sequence of frames → next frames)

## Simple Recurrent Neural Networks

Simple RNNs are the most basic form of recurrent neural networks. They function similarly to feedforward networks but include connections that loop back, enabling information from previous steps to influence current processing.

### Operation
At each timestep, the model:
- Takes in a new input
- Uses its current hidden state (memory of previous inputs)
- Produces both an output and an updated hidden state

### Limitations
Simple RNNs often face the vanishing gradient problem during backpropagation through time:
- Gradients decrease exponentially with small weights
- Early inputs in a sequence are gradually "forgotten"
- Training becomes ineffective for long sequences

## Long Short-Term Memory Networks (LSTMs)

LSTMs are advanced RNNs designed to better handle long sequences through dual memory channels.

### Key Components
1. **State Types**:
   - Cell State: Long-term memory preservation
   - Hidden State: Working memory for immediate processing

2. **Gates**:
   - Forget gate: Removes information from cell state
   - Input gate: Stores new information
   - Output gate: Controls cell state output

### Advantages
- Maintains important information for longer periods
- Solves vanishing gradient problem
- Selective memory updates through gating mechanisms

## Data Preparation

### Text Processing Steps
1. **Loading the Text**:
   - Standardization:
     - Convert to lowercase
     - Standardize line endings and spacing
   - Create corpus for training and inference

2. **Character-to-Index Mapping**:
   - Create vocabulary of unique characters
   - Assign unique integers to each character
   - Enable bidirectional conversion

### Index Representation Methods

#### One-Hot Encoding
- Converts index to binary vector
- Sparse representation
- No implied relationships between characters

#### Embeddings
- Converts index to learned dense vector
- Learnable representation
- Positions similar characters close together
- More memory efficient than one-hot encoding

### Training Data Creation
- Uses sliding window approach
- Fixed-length sequences
- Input shape: (NUM_SEQUENCES, SEQUENCE_LEN)
- Target shape: (NUM_SEQUENCES, 1)

## Training Process
- Target data transformed to one-hot vectors
- Model outputs probability distribution over characters
- Uses categorical cross-entropy loss

## Generation Process

### Key Components
1. **Seed Selection**:
   - Random seed sequence from corpus
   - Provides initial context

2. **Prediction Loop**:
   - Model predicts probability distribution
   - Samples next character
   - Slides window forward

3. **Temperature Parameter**:
   - Controls prediction randomness:
     - less than 0.5: More conservative
     - more than 1.0: More diverse
     - 0.5: Balanced approach