# Expanded local asset check

Searched filenames under /Users/uyen/Downloads/BONEVQA and /Users/uyen/Downloads/BoneMedVQA, including hidden entries visible to rg, excluding Git internals, site-packages and Python bytecode. Queries covered adj_matrix, organ_disease_info, M3AE checkpoint extensions, SAM-Med2D, Vicuna, SmolVLM and RoBERTa.

Matches were four OpenCLIP RoBERTa configuration JSON files and the FAVP Vicuna YAML configuration. No requested native weights or 577-node graph asset filenames were found in that search. Configuration files are not model weights. This is a scoped filename search, not proof of whole-machine absence or absence under alternative filenames.

The pinned LaPA run_scripts/train.sh explicitly composes clip16 and text_roberta with image_size=384, tokenizer=download/roberta-base, load_path=download/pretrained/m3ae.ckpt, and dataset-specific ans_tokenizer.pt/ans_tokenizer_dict.pkl. clip16 selects ViT-B/16 and text_roberta sets vocabulary50265/embedding768; the launch overrides clip16's default224 image size to384. It requests one GPU and microbatch16. The RAD launch explicitly sets max_epoch20; no such epoch budget is authorized for the current BoneVQA study.

The original dataset answer-token files cannot be silently reused as a BoneVQA TRAIN bank. Their preprocessing must be identified and reproduced on the sealed TRAIN vocabulary, with that dataset adaptation disclosed. The graph adjacency/node-token files must remain mutually aligned.

No model execution, package installation, asset download or new quality training took place. All completed pilot artifacts remain unchanged. Native reproduction remains blocked on verified asset availability/provenance and a compatible controlled runtime.
