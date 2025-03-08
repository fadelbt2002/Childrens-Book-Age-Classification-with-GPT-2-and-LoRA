import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Model
from peft import LoraConfig, get_peft_model

class GPT2AgeClassifier(nn.Module):
    """
    GPT-2 based age classification model with LoRA adaptation
    """
    def __init__(self, base_model, hidden_size=768, num_classes=7):
        """
        Initialize the model
        
        Args:
            base_model (GPT2Model): Pre-trained GPT-2 base model
            hidden_size (int): Size of hidden layer
            num_classes (int): Number of age classification categories
        """
        super().__init__()
        self.base_model = base_model
        self.base_model.config.output_hidden_states = True
        
        # Multi-layer Perceptron for classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
        
        Returns:
            torch.Tensor: Classification logits
        """
        # Get model outputs
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # Extract hidden states from last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Extract [cls] token representation (last token)
        cls_output = hidden_states[:, -1, :]
        
        # Classification
        logits = self.mlp(cls_output)
        
        return logits

def create_lora_model(model, lora_config=None):
    """
    Apply LoRA adaptation to the model
    
    Args:
        model (GPT2Model): Base GPT-2 model
        lora_config (LoraConfig, optional): LoRA configuration
    
    Returns:
        PeftModel: Model with LoRA adapters
    """
    if lora_config is None:
        lora_config = LoraConfig(
            r=8,  # Rank of decomposition
            lora_alpha=32,  # Scaling factor
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none"
        )
    
    return get_peft_model(model, lora_config)

def load_base_model(model_name="gpt2", random_init=True):
    """
    Load base GPT-2 model
    
    Args:
        model_name (str): Hugging Face model name
        random_init (bool): Whether to use random initialization
    
    Returns:
        GPT2Model: Base model
    """
    if random_init:
        # Use configuration for randomized weights
        config = GPT2Model.config_class.from_pretrained(model_name)
        config.use_cache = False
        model = GPT2Model(config)
    else:
        # Use pre-trained weights
        model = GPT2Model.from_pretrained(model_name)
    
    return model

def initialize_model(
    tokenizer, 
    num_classes=7, 
    random_init=True, 
    apply_lora=True
):
    """
    Initialize complete model pipeline
    
    Args:
        tokenizer (GPT2Tokenizer): Tokenizer
        num_classes (int): Number of age classification categories
        random_init (bool): Whether to use random initialization
        apply_lora (bool): Whether to apply LoRA adaptation
    
    Returns:
        tuple: Initialized model and updated tokenizer
    """
    # Add [cls] token to tokenizer
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[cls]"]
    })
    
    # Load base model
    base_model = load_base_model(random_init=random_init)
    
    # Resize token embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA if specified
    if apply_lora:
        base_model = create_lora_model(base_model)
    
    # Create classifier model
    model = GPT2AgeClassifier(base_model, num_classes=num_classes)
    
    return model, tokenizer
