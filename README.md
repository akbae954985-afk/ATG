# ATG
Official code for our paper "**An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction**" which will be published at AAAI 2024.

## Recent Updates
- ✅ Added train config file (`config.yaml`) with comprehensive configuration options
- ✅ Removed dependency on AllenNLP - now uses pure PyTorch implementations
- ✅ Updated model paths to use Hugging Face Hub model names for better accessibility
- ✅ Added Persian language support using PartAI/TookaBERT-Base model
- ✅ Enhanced evaluation logging with console output and detailed metrics
- ✅ Replaced Flair dependency with direct Transformers implementation for better compatibility

## Usage

### Training with Configuration File (Persian Language)
```bash
python train.py --config config.yaml --log_dir ./logs
```

### Training with Command Line Arguments (Legacy)
```bash
# Persian model (default)
python train.py --data_file dataset/scierc.pkl --model_name tookabert --log_dir ./logs

# English scientific model
python train.py --data_file dataset/scierc.pkl --model_name scibert_cased --log_dir ./logs

# Other supported models
python train.py --data_file dataset/scierc.pkl --model_name bert --log_dir ./logs
```

### Supported Models
- **Persian**: `tookabert` (PartAI/TookaBERT-Base) - Default
- **English Scientific**: `scibert_cased` (allenai/scibert_scivocab_cased)
- **Multilingual**: `bert`, `roberta`, `deberta`
- **Arabic**: `arabert`
- **And more**: See `config.yaml` for full list

## Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📁 File Structure Guide

### Core Training Files

#### `train.py` (Main Training Script)
- **Purpose**: Main entry point for training the ATG model
- **Key Functions**:
  - `train()`: Main training loop with evaluation and checkpointing
  - `evaluate()`: Model evaluation wrapper
  - `load_config()`: Load YAML configuration files
  - `config_to_args()`: Convert config to argparse namespace
  - `create_parser()`: Command-line argument parser
- **Usage**: 
  ```bash
  python train.py --config config.yaml --log_dir ./logs
  python train.py --model_name tookabert --data_file dataset/scierc.pkl --log_dir ./logs
  ```

#### `config.yaml` (Configuration File)
- **Purpose**: YAML configuration file containing all training parameters
- **Sections**:
  - Data Configuration (dataset paths)
  - Model Configuration (architecture parameters)
  - Training Configuration (epochs, batch sizes)
  - Optimizer Configuration (learning rates)
  - Logging Configuration
  - Model Paths (Hugging Face model mappings)

#### `model.py` (Main Model Architecture)
- **Purpose**: Defines the `IeGenerator` class - the main ATG model
- **Key Components**:
  - `IeGenerator`: Main autoregressive text-to-graph model
  - `MLP()`: Multi-layer perceptron utility function
- **Key Methods**:
  - `compute_token_embeddings()`: Process input tokens through BERT + LSTM
  - `get_splits_queries_out_emb()`: Extract span representations and queries
  - `get_transformer_input()`: Prepare inputs for transformer decoder
  - `forward()`: Main forward pass for training

### Data Processing Files

#### `preprocess.py` (Data Loading & Preprocessing)
- **Purpose**: Dataset class for loading and preprocessing graph IE data
- **Key Classes**:
  - `GraphIEData`: PyTorch Dataset for graph information extraction
- **Key Methods**:
  - `create_seq()`: Convert raw data to autoregressive sequences
  - `create_input()`: Prepare model inputs (tokens, spans, masks)
  - `__getitem__()`: Dataset item retrieval with augmentation

#### `generate.py` (Evaluation & Generation)
- **Purpose**: Model evaluation and sequence generation
- **Key Classes**:
  - `Evaluator`: Handles model evaluation on test/validation sets
- **Key Methods**:
  - `evaluate()`: Run full evaluation pipeline
  - `get_entities()`: Extract entities from generated sequences
  - `get_relations()`: Extract relations from generated sequences
  - `evaluate_all_with_loader()`: Batch evaluation with data loader

#### `metric.py` (Evaluation Metrics)
- **Purpose**: Precision, recall, F1 score calculations for entities and relations
- **Key Functions**:
  - `extract_tp_actual_correct()`: Extract true positives, actuals, and corrects
  - `compute_prf()`: Compute precision, recall, F1 scores
  - `merge_graphs()`: Merge multiple prediction graphs
- **Evaluation Types**: Supports strict/relaxed and symmetric/asymmetric evaluation

### Model Architecture Components

#### `layers/` Directory

##### `layers/base.py` (Base Classes)
- **Purpose**: Base class for joint relation extraction models
- **Key Classes**:
  - `BaseJointRE`: Base class with common functionality
- **Key Methods**:
  - `create_dataloader()`: Create PyTorch DataLoader with custom collation
  - `collate_fn()`: Custom batch collation function

##### `layers/token_embedding_robust.py` (Token Representations)
- **Purpose**: Token embedding layer using transformers directly (BERT, etc.)
- **Key Classes**:
  - `TokenRep`: Direct transformer implementation with query support
- **Key Features**:
  - Uses Hugging Face Transformers directly (no Flair dependency)
  - Supports learnable query embeddings
  - Handles subtoken pooling strategies
  - Better version compatibility and stability

##### `layers/span_embedding.py` (Span Representations)
- **Purpose**: Various span representation methods
- **Key Classes**:
  - `SpanRepLayer`: Factory class for different span representation methods
  - `SpanEndpoints`: Endpoint-based span representation
  - `SpanAttention`: Self-attention-based span representation  
  - `SpanConv`: Convolutional span representation
  - `SpanMarker`: Marker-based span representation
  - And many more span representation variants

##### `layers/lstm_encoder.py` (LSTM Encoder)
- **Purpose**: Custom PyTorch LSTM sequence-to-sequence encoder
- **Key Classes**:
  - `LstmSeq2SeqEncoder`: Bidirectional LSTM encoder (replaces AllenNLP)
- **Features**:
  - Supports sequence masking
  - Configurable bidirectional processing
  - Compatible with packed sequences

##### `layers/span_extractors.py` (Span Extractors)
- **Purpose**: Custom span extraction methods (replaces AllenNLP)
- **Key Classes**:
  - `EndpointSpanExtractor`: Extract spans by start/end positions
  - `SelfAttentiveSpanExtractor`: Self-attention over span tokens
  - `BidirectionalEndpointSpanExtractor`: Bidirectional endpoint extraction
- **Features**: Multiple combination strategies (x,y / x*y / x+y / x-y)

##### `layers/structure.py` (Structural Components)
- **Purpose**: Positional embeddings and structural encodings
- **Key Functions**:
  - `create_position_code_sep()`: Create position codes for graph sequences
- **Key Classes**:
  - `PosEmbCodeSep`: Positional embedding with separator tokens

#### `trans_enc.py` (Transformer Decoder)
- **Purpose**: Autoregressive transformer decoder
- **Key Classes**:
  - `TransDec`: Transformer decoder for sequence generation
- **Key Functions**:
  - `generate_square_subsequent_mask()`: Create causal attention mask
- **Features**:
  - Supports both encoder-only and encoder-decoder modes
  - Optional cross-attention
  - Positional embedding integration

### Utility Files

#### `save_load.py` (Model Persistence)
- **Purpose**: Model saving and loading utilities
- **Key Functions**:
  - `save_model()`: Save model weights and configuration
  - `load_model()`: Load model from saved checkpoint
- **Features**: Preserves model architecture parameters for proper reconstruction

#### `requirements.txt` (Dependencies)
- **Purpose**: Python package dependencies
- **Key Packages**:
  - `torch`: Deep learning framework
  - `transformers`: Hugging Face transformers (direct implementation)
  - `tokenizers`: Fast tokenization
  - `PyYAML`: Configuration file support
  - `tqdm`: Progress bars
  - `numpy`: Numerical operations

## 🔄 Training Pipeline Flow

1. **Configuration** (`config.yaml` → `train.py`): Load training parameters
2. **Data Loading** (`preprocess.py`): Load and preprocess datasets
3. **Model Creation** (`model.py` + `layers/`): Initialize ATG model architecture
4. **Training Loop** (`train.py`): 
   - Forward pass through model
   - Loss computation
   - Backpropagation and optimization
   - Periodic evaluation (`generate.py` + `metric.py`)
   - Model checkpointing (`save_load.py`)
5. **Evaluation** (`generate.py` + `metric.py`): Compute precision, recall, F1 scores

## 🎯 Key Model Components

- **Token Encoder**: BERT/TookaBERT → BiLSTM → Projections
- **Span Encoder**: Multiple span representation methods (endpoints, attention, convolution)
- **Autoregressive Decoder**: Transformer decoder for sequence generation
- **Loss Function**: Cross-entropy on next token prediction
- **Evaluation**: Entity and relation extraction metrics


## Citation

```bibtex
@misc{urchade2024autoregressive,
      title={An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction}, 
      author={Zaratiana Urchade and Nadi Tomeh and Pierre Holat and Thierry Charnois},
      year={2024},
      eprint={2401.01326},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
