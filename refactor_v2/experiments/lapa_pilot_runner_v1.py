"""Five prospectively specified routes; no automatic training entry point."""
import torch
from experiments.lapa_integrated_gate_v1 import Runner

ARMS = {'carrier':('train',0.), 'train_cs0':('train',0.),
        'train_cs1':('train',.1), 'random_cs0':('random',0.),
        'random_cs1':('random',.1)}


class PilotRunner(Runner):
    def __init__(self, arm):
        if arm not in ARMS: raise ValueError(arm)
        super().__init__()
        self.arm = arm
        self.bank_mode,self.eta = ARMS[arm]
        with torch.no_grad():
            raw = self.text.model.get_input_embeddings()(self.bank_ids).detach()
            generator = torch.Generator(device='cpu').manual_seed(20261025)
            random = torch.randn(raw.shape,generator=generator,dtype=raw.dtype)
            random = random/random.norm(dim=1,keepdim=True)*raw.norm(dim=1,keepdim=True)
        self.register_buffer('random_bank',random)
        assert torch.allclose(random.norm(dim=1),raw.norm(dim=1),atol=1e-6,rtol=1e-6)
        # Graph modules are constructed for source compatibility but unused.
        for module in (self.reference.prompt.PF,self.reference.prompt.graph_reduction,
                       self.reference.prompt.combined_reduction):
            module.requires_grad_(False)
        if arm=='carrier': self.reference.prompt.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        # Shared frozen encoder dropout remains off; visual LoRA still trains.
        self.visual.eval(); self.text.eval()
        return self

    def forward(self,images,questions):
        encoded=self.text(questions,torch.device('cpu'))
        tokens=self.visual(self.visual.preprocess(images,torch.device('cpu')))['tokens']
        bank=(self.random_bank if self.bank_mode=='random' else
              self.text.model.get_input_embeddings()(self.bank_ids))
        result=self.reference(encoded['tokens'],tokens,encoded['mask'],bank,
                              include_prompt=self.arm!='carrier')
        return self.head(result['features']),result['generated_mean']

    def optimizer(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                 lr=1e-4,weight_decay=.01)
