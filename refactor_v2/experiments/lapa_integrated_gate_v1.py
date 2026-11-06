"""Two disposable real-TRAIN updates and fresh-process public-input replay."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments.check_lapa_real_features_v1 import REPORTS, sha
from experiments.lapa_interleaved_reference_v1 import InterleavedReference
from experiments.lapa_reference_endpoint_v1 import ReferenceAnswerHead, reference_loss
from src.models.encoders import VisualEncoder, TextEncoder
from src.data import build_dataset, collate_fn
from src.train import build_split
from src.utils import normalize_answer

OUT = REPORTS/'lapa_integrated_gate_v1'


def digest(tensors):
    h = hashlib.sha256()
    for name, tensor in tensors:
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Runner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualEncoder('biomedclip',768,lora_rank=4,image_size=224)
        self.text = TextEncoder(self.visual,768,max_len=48)
        self.vocab = json.loads((REPORTS/'step4_localization_v3_1/vocab.json').read_text())
        ids = [t for answer in self.vocab for t in self.text.tokenizer.encode(answer,add_special_tokens=True)]
        self.register_buffer('bank_ids',torch.tensor(ids))
        self.reference = InterleavedReference(len(ids))
        self.head = ReferenceAnswerHead(768,len(self.vocab))

    def forward(self, images, questions):
        # Public signature intentionally excludes answers, boxes and labels.
        encoded = self.text(questions,torch.device('cpu'))
        tokens = self.visual(self.visual.preprocess(images,torch.device('cpu')))['tokens']
        bank = self.text.model.get_input_embeddings()(self.bank_ids)
        result = self.reference(encoded['tokens'],tokens,encoded['mask'],bank)
        return self.head(result['features']),result['generated_mean']


def batch_data():
    cfg = json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    ds = build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    indices = json.loads((REPORTS/'vnext_local_tokens_v1/gate_indices.json').read_text())[:2]
    return collate_fn([ds[i] for i in indices]),[normalize_answer(ds[i]['answer']) for i in indices]


def main():
    torch.set_num_threads(8); torch.manual_seed(42)
    replay = '--replay' in sys.argv
    if not replay: OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    binding = json.loads((REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json').read_text())
    for model in binding['models']:
        for file in model['files']: assert sha(Path(file['resolved_path']))==file['sha256']
    runner = Runner().eval()
    batch, answers = batch_data()
    if replay:
        saved = torch.load(OUT/'checkpoint.pt',map_location='cpu')
        runner.load_state_dict(saved['state'],strict=True)
        with torch.no_grad(): logits,_ = runner(batch['images'],batch['questions'])
        assert torch.equal(logits,saved['logits'])
        (OUT/'replay.json').write_text(json.dumps({'strict_full_state':True,'exact_logits':True}))
        return
    checkpoint = REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release = json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(checkpoint)==release['dependencies'][str(checkpoint)]
    state = torch.load(checkpoint,map_location='cpu')['state']
    supplied = {k[7:]:v for k,v in state.items() if k.startswith('visual.')}
    loaded = runner.visual.load_state_dict(supplied,strict=False)
    assert not loaded.unexpected_keys
    assert all(torch.equal(v,runner.visual.state_dict()[k]) for k,v in supplied.items())
    del state,supplied
    # Fixed TRAIN-vocabulary-wide padding length, including special/pad tokens.
    length = max(len(runner.text.tokenizer.encode(a,add_special_tokens=True)) for a in runner.vocab)
    target_ids = runner.text.tokenizer(answers,padding='max_length',max_length=length,truncation=False,return_tensors='pt')['input_ids']
    with torch.no_grad(): words = runner.text.model.get_input_embeddings()(target_ids)
    targets = F.one_hot(torch.tensor([runner.vocab.index(a) for a in answers]),len(runner.vocab)).float()
    frozen_before = digest((n,p) for n,p in runner.named_parameters() if not p.requires_grad)
    groups = {'visual_lora':[(n,p) for n,p in runner.visual.named_parameters() if 'lora_' in n],
              'GM':list(runner.reference.prompt.generation.named_parameters()),
              'LF':list(runner.reference.prompt.LF.named_parameters()),
              'crossmodal':list(runner.reference.text_layers.named_parameters()),
              'head':list(runner.head.named_parameters())}
    initial = {name:digest(params) for name,params in groups.items()}
    optimizer = torch.optim.SGD([p for p in runner.parameters() if p.requires_grad],lr=1e-4)
    steps=[]
    for step in range(2):
        tick=time.perf_counter(); optimizer.zero_grad()
        logits,generated=runner(batch['images'],batch['questions'])
        losses=reference_loss(logits,targets,generated,words,eta=.1)
        if step==0:
            probes=[groups['visual_lora'][-1][1],groups['head'][0][1]]
            routed=torch.autograd.grad(losses['consistency_sum'],probes,allow_unused=True,retain_graph=True)
            assert all(g is None for g in routed)
        losses['total'].backward()
        gradients={}
        for name,params in groups.items():
            gs=[p.grad for _,p in params if p.grad is not None]
            assert gs and all(torch.isfinite(g).all() for g in gs)
            gradients[name]=max(g.abs().max().item() for g in gs)
            assert gradients[name]>0,name
        optimizer.step()
        steps.append({'loss':losses['total'].item(),'seconds':time.perf_counter()-tick,'gradients':gradients})
    changed={name:digest(params)!=initial[name] for name,params in groups.items()}
    assert all(changed.values())
    assert frozen_before==digest((n,p) for n,p in runner.named_parameters() if not p.requires_grad)
    with torch.no_grad(): logits,_=runner(batch['images'],batch['questions'])
    torch.save({'state':runner.state_dict(),'logits':logits},OUT/'checkpoint.pt')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],check=True)
    receipt={'scope':'Two real TRAIN records, full-width reference head and loss; eval-mode dropout with gradients, SGD technical updates only. Not a quality training configuration.',
             'image_ids':batch['image_ids'],'target_padding_length':length,'target_padding_mean':'unmasked including special/pad tokens',
             'steps':steps,'updated_groups':changed,'frozen_parameters_unchanged':True,
             'consistency_no_visual_or_head_route':True,'public_inputs':['images','questions'],
             'reload':json.loads((OUT/'replay.json').read_text()),'total_seconds':time.perf_counter()-started,
             'quality_release':False,'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))


if __name__=='__main__':
    try: main()
    except Exception as error:
        if OUT.exists():
            with (OUT/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f: json.dump({'error':repr(error)},f)
        raise
