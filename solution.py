#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(1024, d_model)  # Assuming a maximum sequence length of 1024

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        return self.token_embeddings(x) + self.position_embeddings(positions)


# In[35]:


import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.W_q(q), batch_size)
        k = self.split_heads(self.W_k(k), batch_size)
        v = self.split_heads(self.W_v(v), batch_size)

        # Calculate the attention scores
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        
        # Scale matmul_qk
        d_k = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(d_k)

        # Apply the mask (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax to get the attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Multiply by values
        scaled_attention = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(original_size_attention)

        return output


# In[36]:


class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(nn.functional.relu(self.linear1(x)))


# In[44]:


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_input = self.layernorm1(x)
        attn_output = self.mha(attn_input, attn_input, attn_input)  # Pre-LN
        x = x + self.dropout1(attn_output)  # Apply residual connection
        ffn_input = self.layernorm2(x)
        ffn_output = self.ffn(ffn_input)  # Pre-LN
        x = x + self.dropout2(ffn_output)  # Apply residual connection
        return x


# In[45]:


class GPT2(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, vocab_size, d_ff, dropout_rate):
        super(GPT2, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        logits = self.final_layer(x)
        return logits


# In[46]:


batch_size = 1
seq_length = 10  # or another sequence length of your choice
vocab_size = 50257  # typical GPT-2 vocab size

# Generate a random sequence of token IDs
sample_input = torch.randint(vocab_size, (batch_size, seq_length))


# In[47]:


num_layers = 12
d_model = 768
num_heads = 12
d_ff = 3072
dropout_rate = 0.1

model = GPT2(num_layers, d_model, num_heads, vocab_size, d_ff, dropout_rate)


# In[48]:


output = model(sample_input)
print(output.shape)  # should be [batch_size, seq_length, vocab_size]


# In[49]:


import torch
import random

# Assuming your GPT2 model and other classes (EmbeddingLayer, etc.) are defined above

def mock_tokenizer(text, vocab_size):
    """
    Convert text to a list of token ids.
    This is a placeholder and should be replaced with a proper tokenizer.
    """
    return [random.randint(0, vocab_size-1) for _ in text]

def generate_text(model, initial_text, vocab_size, max_length=50):
    """
    Generate text using the untrained model.
    """
    model.eval()
    input_ids = torch.tensor([mock_tokenizer(initial_text, vocab_size)], dtype=torch.long)
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[0, -1, :]
            next_token_id = torch.argmax(predictions).unsqueeze(0).unsqueeze(0)
            input_ids = next_token_id
            generated = torch.cat((generated, next_token_id), dim=1)

    return generated[0].tolist()

# Initialize your model
vocab_size = 50257  # typical GPT-2 vocab size
num_layers = 12
d_model = 768
num_heads = 12
d_ff = 3072
dropout_rate = 0.1

model = GPT2(num_layers, d_model, num_heads, vocab_size, d_ff, dropout_rate)

# Generate text
initial_text = "Hello world"
generated_token_ids = generate_text(model, initial_text, vocab_size)
print("Generated Token IDs:", generated_token_ids)

# Convert token IDs back to text
# This requires a reverse mapping from the mock tokenizer
# For now, it will just print token IDs


# In[ ]:




