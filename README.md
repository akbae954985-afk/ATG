# ATG
Official code for our paper "**An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction**" which will be published at AAAI 2024.

## Recent Updates
- ✅ Added train config file (`config.yaml`) with comprehensive configuration options
- ✅ Removed dependency on AllenNLP - now uses pure PyTorch implementations

## Usage

### Training with Configuration File
```bash
python train.py --config config.yaml --log_dir ./logs
```

### Training with Command Line Arguments (Legacy)
```bash
python train.py --data_file dataset/scierc.pkl --model_name scibert_cased --log_dir ./logs
```

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
