# src/models/conformer_asr.py
import torch
import torch.nn as nn
from torchaudio.models import Conformer

class SelfHostedASRConformer(nn.Module):
    def __init__(self, num_classes=5000, input_dim=80, d_model=256, num_heads=4, num_layers=12):
        """
        Pure PyTorch Conformer-CTC Speech-to-Text Model.
        - input_dim: 80 log-mel filterbank channels
        - d_model: internal transformer representation size
        - num_classes: vocabulary size (SentencePiece subwords)
        """
        super().__init__()
        
        # Linear projection from filterbank dim to model dim
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Torchaudio native conformer backbone
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=1024,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        
        # CTC classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lengths):
        # x shape: (batch_size, time_steps, 80)
        x = self.input_projection(x)
        out, out_lengths = self.conformer(x, lengths)
        logits = self.classifier(out)
        return self.log_softmax(logits), out_lengths

if __name__ == "__main__":
    # Test on your H100 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfHostedASRConformer().to(device)
    
    # Dummy batch: 4 audio clips, 200 time frames, 80 mel bins
    dummy_input = torch.randn(4, 200, 80).to(device)
    lengths = torch.tensor([200, 180, 150, 120]).to(device)
    
    with torch.no_grad():
        log_probs, _ = model(dummy_input, lengths)
    print(f"Conformer successfully executed on: {device}")
    print(f"Output shape: {log_probs.shape} (Batch, Time, Vocab)")