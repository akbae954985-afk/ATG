import logging
import os
import pickle
import yaml

import torch
from tqdm import tqdm

        if (step + 1) % save_interval == 0:
            # Create log directory
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Evaluate
            logging.info("Starting evaluation...")
            metric_dict, metrics = evaluate(model, eval_loader)
            
            # Print evaluation results to console
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS - Step {step + 1}")
            print(f"{'='*60}")
            print(metrics)
            print(f"{'='*60}\n")
            
            # Log evaluation results
            logging.info(f"Evaluation at step {step + 1}:")
            logging.info(f"\n{metrics}")

            # Save metrics to file if log_dir is specified
            if log_dir:
                with open(os.path.join(log_dir, 'log_metrics.txt'), 'a') as f:
                    f.write(f'{description}\n\n')
                    f.write(f'{metrics}\n\n\n')

            # current f1 for Strict + not Symetric evaluation
            current_f1 = float(metric_dict["Strict + not Symetric"]["f_score"])
            
            logging.info(f"Current F1: {current_f1:.4f}, Best F1: {best_f1:.4f}")

            if current_f1 > best_f1:
                logging.info(f"New best F1 score: {current_f1:.4f} (previous: {best_f1:.4f})")
                
                if log_dir:
                    # save current best model
                    current_path = os.path.join(log_dir, f'model_{step + 1}_{current_f1:.4f}.pt')
                    save_model(model, current_path)
                    logging.info(f"Saved new best model: {current_path}")

                    if best_path is not None and os.path.exists(best_path):
                        os.remove(best_path)
                        logging.info(f"Removed previous best model: {best_path}")
                    best_path = current_path
                    
                best_f1 = current_f1

            model.train()

    # Final evaluation at the end of training
    logging.info("Training completed. Running final evaluation...")
    model.eval()
    metric_dict, metrics = evaluate(model, eval_loader)
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(metrics)
    print(f"{'='*60}\n")
    
    logging.info(f"Final evaluation results:\n{metrics}")
    
    if log_dir:
        with open(os.path.join(log_dir, 'final_metrics.txt'), 'w') as f:
            f.write(f"Final evaluation results:\n\n{metrics}\n")
        logging.info(f"Final metrics saved to {os.path.join(log_dir, 'final_metrics.txt')}")
    
    final_f1 = float(metric_dict["Strict + not Symetric"]["f_score"])
    logging.info(f"Final F1 score: {final_f1:.4f}")
    
    return best_path, best_f1


MODELS = {
    "spanbert": "SpanBERT/spanbert-base-cased",
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/roberta-base",
    "scibert": "allenai/scibert_scivocab_uncased",
    "arabert": "aubmindlab/bert-base-arabert",
    "bertlarge": "google-bert/bert-large-cased",
    "scibert_cased": "allenai/scibert_scivocab_cased",
    "albert": "albert/albert-xxlarge-v2",
    "spanbertlarge": "SpanBERT/spanbert-large-cased",
    "t5-s": "google-t5/t5-small",
    "t5-m": "google-t5/t5-base",
    "t5-l": "google-t5/t5-large",
    "deberta": "microsoft/deberta-v3-large",
    "tookabert": "PartAI/TookaBERT-Base"
}


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """Convert config dictionary to argparse.Namespace."""
    # Handle model paths
    if 'model_paths' in config and config['model_name'] in config['model_paths']:
        model_path = config['model_paths'][config['model_name']]
    else:
        # Fallback to old MODELS dict if not in config
        model_path = MODELS.get(config['model_name'], config['model_name'])
    
    # Create namespace with all config values
    args_dict = {k: v for k, v in config.items() if k != 'model_paths'}
    args_dict['model_path'] = model_path
    
    return argparse.Namespace(**args_dict)


# #training_arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--data_file', type=str, default='dataset/scierc.pkl')
    parser.add_argument('--model_name', type=str, default='tookabert')
    parser.add_argument('--max_width', type=int, default=14)
    parser.add_argument('--num_prompts', type=int, default=5)
    parser.add_argument('--hidden_transformer', type=int, default=512)
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        config = load_config(args.config)
        args = config_to_args(config)
        model_path = args.model_path
    else:
        model_path = MODELS[args.model_name]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Open the file
    logging.info(f"Loading dataset from: {args.data_file}")
    with open(args.data_file, 'rb') as f:
        datasets = pickle.load(f)

    class_to_id = datasets['span_to_id']  # entity to id mapping
    rel_to_id = datasets['rel_to_id']  # relation to id mapping
    rel_to_id["stop_entity"] = len(rel_to_id)  # add a new relation for stop entity
    
    logging.info(f"Dataset loaded successfully:")
    logging.info(f"  - Number of entity types: {len(class_to_id)}")
    logging.info(f"  - Number of relation types: {len(rel_to_id)}")
    logging.info(f"  - Training samples: {len(datasets['train']) if 'train' in datasets else 0}")
    logging.info(f"  - Dev samples: {len(datasets['dev']) if 'dev' in datasets else 0}")
    logging.info(f"  - Test samples: {len(datasets.get('test', []))}")

    model = IeGenerator(
        class_to_id, rel_to_id, model_name=model_path, max_width=args.max_width,
        num_prompts=args.num_prompts, hidden_transformer=args.hidden_transformer,
        num_transformer_layers=args.num_transformer_layers, attention_heads=args.attention_heads,
        span_mode=args.span_mode, use_pos_code=args.use_pos_code, p_drop=args.p_drop, cross_attn=args.cross_attn
        {'params': model.embed_proj.parameters(), 'lr': args.lr_others},
    ])

    best_model_path, best_f1 = train(
        model=model, optimizer=optimizer, train_data=datasets['train'], eval_data=datasets['dev'],
        train_batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        n_epochs=args.n_epochs, n_steps=args.n_steps, warmup_ratio=args.warmup_ratio,
        max_num_samples=args.max_num_samples,
        save_interval=args.save_interval, log_dir=args.log_dir
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Best F1 score achieved: {best_f1:.4f}")
    if best_model_path:
        print(f"💾 Best model saved at: {best_model_path}")
    else:
        print(f"⚠️  No model was saved (log_dir not specified)")
    print(f"📁 Logs and metrics saved in: {args.log_dir}")
    
    logging.info(f"Training session completed. Best F1: {best_f1:.4f}")
