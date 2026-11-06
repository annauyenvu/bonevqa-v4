"""Shared-backbone PF reconstruction; native LaPA assets are not implied."""
import torch
from experiments.lapa_pilot_runner_v1 import PilotRunner
from experiments.pf_slake_reference_v1 import SlakeReferencePF


class PFRunner(PilotRunner):
    def __init__(self, mode='real'):
        if mode not in ('real', 'rewired', 'bypass'):
            raise ValueError(mode)
        super().__init__('train_cs1')
        self.pf_mode = mode
        # Token IDs MUST belong to the actual PubMedBERT embedding table.
        # The standalone RoBERTa-tokenizer check is not used for this port.
        self.slake_pf = SlakeReferencePF(self.text.tokenizer, 768)
        self.reference.prompt.prompt_reduction.requires_grad_(False)

    def forward(self, images, questions):
        encoded = self.text(questions, torch.device('cpu'))
        image = self.visual(self.visual.preprocess(images, torch.device('cpu')))['tokens']
        embedding = self.text.model.get_input_embeddings()
        result = self.reference(encoded['tokens'], image, encoded['mask'], embedding(self.bank_ids))
        assert result['trace'][-2:] == ['MM6', 'LF6']
        summary = self.slake_pf(result['lf_prompts'], embedding, self.pf_mode)
        return self.head(result['carrier'] + summary), result['generated_mean']
