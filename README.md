# ATG
Official code for our paper "**An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction**" which will be published at AAAI 2024.

## Recent Updates
- ✅ Added train config file (`config.yaml`) with comprehensive configuration options
- ✅ Removed dependency on AllenNLP - now uses pure PyTorch implementations
- ✅ Updated model paths to use Hugging Face Hub model names for better accessibility
- ✅ Added Persian language support using PartAI/TookaBERT-Base model

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
