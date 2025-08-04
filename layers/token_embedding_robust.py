import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel


class TokenRep(nn.Module):
    """
    Robust token representation using Transformers directly instead of Flair.
    This avoids issues with Flair's private methods and version compatibility.
    """

    def __init__(self, num_queries=40, model_name="bert-base-cased", fine_tune=True, subtoken_pooling="first"):
        super().__init__()

        self.model_name = model_name
        self.fine_tune = fine_tune
        self.subtoken_pooling = subtoken_pooling
        self.num_queries = num_queries

        # Load tokenizer and model directly from transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer_model = AutoModel.from_pretrained(model_name)
        
        # Set fine-tuning
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        # Hidden size from the transformer model
        self.hidden_size = self.transformer_model.config.hidden_size

        # Query embeddings
        if self.num_queries == 0:
            self.query_embedding = None
        else:
            # Use the same embedding dimension as the transformer model
            embedding_dim = self.transformer_model.get_input_embeddings().embedding_dim
            self.query_embedding = nn.Parameter(torch.randn(num_queries, embedding_dim))
            nn.init.uniform_(self.query_embedding, -0.01, 0.01)

    def forward(self, sentences, seq_lengths):
        """
        Forward pass through the token representation layer.
        
        Args:
            sentences: List of tokenized sentences (list of lists of strings)
            seq_lengths: Tensor of sequence lengths
            
        Returns:
            Dictionary containing queries, embeddings, mask, and cache
        """
        # Convert sentences to strings if they're tokenized
        if isinstance(sentences[0], list):
            text_sentences = [" ".join(sent) for sent in sentences]
        else:
            text_sentences = sentences

        # Tokenize all sentences
        tokenized = self.tokenizer(
            text_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=512  # Reasonable max length
        )
        
        input_ids = tokenized['input_ids'].to(next(self.parameters()).device)
        attention_mask = tokenized['attention_mask'].to(next(self.parameters()).device)
        
        batch_size, max_seq_len = input_ids.shape

        # Handle query embeddings
        if self.query_embedding is not None:
            num_queries = self.query_embedding.size(0)
            
            # Expand query embeddings for the batch
            queries = self.query_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Get input embeddings
            input_embeddings = self.transformer_model.get_input_embeddings()(input_ids)
            
            # Concatenate queries with input embeddings
            input_embeddings = torch.cat([queries, input_embeddings], dim=1)
            
            # Extend attention mask for queries
            query_mask = torch.ones(batch_size, num_queries, device=attention_mask.device)
            attention_mask = torch.cat([query_mask, attention_mask], dim=1)
            
            # Forward through transformer
            outputs = self.transformer_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states from all layers
            hidden_states = outputs.hidden_states  # Tuple of hidden states from each layer
            
            # Use the last layer's hidden states
            last_hidden_state = hidden_states[-1]  # (batch_size, seq_len + num_queries, hidden_size)
            
            # Split queries and token embeddings
            query_embeddings = last_hidden_state[:, :num_queries, :]  # (batch_size, num_queries, hidden_size)
            token_embeddings = last_hidden_state[:, num_queries:, :]   # (batch_size, seq_len, hidden_size)
            
            # Revert attention mask to original tokens only
            token_attention_mask = attention_mask[:, num_queries:]
            
        else:
            # No query embeddings
            outputs = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states
            token_embeddings = hidden_states[-1]  # Last layer
            query_embeddings = torch.zeros(batch_size, 0, self.hidden_size, device=input_ids.device)
            token_attention_mask = attention_mask

        # Handle subtoken pooling (simplified to first subtoken)
        if self.subtoken_pooling == "first":
            # For now, use the token embeddings as is
            # In a more sophisticated implementation, you'd map back to original tokens
            pass

        # Create mask tensor matching the expected format
        # Limit to the actual sequence lengths
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=input_ids.device)
        for i, length in enumerate(seq_lengths):
            actual_length = min(length.item(), max_seq_len)
            mask[i, :actual_length] = True

        # Ensure embeddings match the expected sequence length
        if token_embeddings.size(1) != max_seq_len:
            if token_embeddings.size(1) > max_seq_len:
                token_embeddings = token_embeddings[:, :max_seq_len, :]
            else:
                # Pad if needed
                padding_size = max_seq_len - token_embeddings.size(1)
                padding = torch.zeros(batch_size, padding_size, self.hidden_size, device=token_embeddings.device)
                token_embeddings = torch.cat([token_embeddings, padding], dim=1)

        return {
            'queries': query_embeddings,
            'embeddings': token_embeddings,
            'mask': mask,
            'cache': None  # Not used in this implementation
        }

    def get_embeddings(self, sentences, queries=None):
        """
        Legacy method for compatibility.
        """
        # Convert to the expected format
        if isinstance(sentences[0], str):
            # If sentences are strings, tokenize them
            sentences = [sent.split() for sent in sentences]
        
        seq_lengths = torch.tensor([len(sent) for sent in sentences])
        result = self.forward(sentences, seq_lengths)
        
        if queries is not None:
            return result['embeddings'], result['queries'], result['cache']
        else:
            return result['embeddings'], None, result['cache']
