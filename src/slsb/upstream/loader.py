"""Frozen upstream encoders via a shared interface.

Encoders are always frozen: .eval() + requires_grad_(False). Only the
per-layer weighted-sum + task heads in slsb/tasks/ train.

Loading is via Hugging Face transformers (AutoModel/AutoFeatureExtractor),
which is what the actual probe/CTC training loops in slsb/tasks/ consume.
`--upstream` accepts either one of the short aliases below or any HF repo id
directly (e.g. "facebook/wav2vec2-xls-r-300m").
"""
import torch
from transformers import AutoFeatureExtractor, AutoModel

# Short aliases for the upstreams this benchmark has been validated against.
# Any other HF repo id also works -- these are just convenient shorthands.
UPSTREAM_ALIASES = {
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "mhubert147": "utter-project/mHuBERT-147",
    "wavlm_large": "microsoft/wavlm-large",
}


class Upstream:
    def __init__(self, name: str, device: torch.device):
        repo = UPSTREAM_ALIASES.get(name, name)
        self.name = name
        self.repo = repo
        self.device = device

        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo)

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.num_hidden_states = self.num_layers + 1  # + input embedding layer

    @torch.no_grad()
    def extract(self, waveforms: list):
        """waveforms: list of 1-D float32 numpy arrays at 16kHz.

        Returns:
          hidden_states: (num_hidden_states, B, T', H) float tensor on self.device
          frame_mask: (B, T') bool tensor, True where the frame is real (not padding)
        """
        inputs = self.feature_extractor(
            waveforms, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L+1, B, T', H)

        if attention_mask is not None and hasattr(self.model, "_get_feature_vector_attention_mask"):
            frame_mask = self.model._get_feature_vector_attention_mask(
                hidden_states.shape[2], attention_mask
            ).bool()
        else:
            frame_mask = torch.ones(hidden_states.shape[1], hidden_states.shape[2], dtype=torch.bool, device=self.device)

        return hidden_states, frame_mask


def load_upstream(name: str, device: torch.device) -> Upstream:
    return Upstream(name, device)
