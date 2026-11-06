"""Verify concrete training config bindings without loading model weights."""
import json
from pathlib import Path

from src.data import build_dataset
from src.protocol import assert_disjoint_splits, validate_prompt_protocol_config
from src.train import build_split
from src.utils import load_config


def main():
    root = Path(__file__).resolve().parents[2]
    results = {}
    for name in ('fold0_b_sealed_v3', 'pmc_sealed_generative_v3'):
        config = load_config(str(root / 'refactor_v2/configs' / (name + '.yaml')))
        validate_prompt_protocol_config(config)
        data = config['data']
        train = build_split(build_dataset, data['dataset'], data['train_split'], None, True, data['loader_kwargs'])
        validation = build_split(build_dataset, data['dataset'], data['val_split'], None, True, data['loader_kwargs'])
        assert_disjoint_splits(train=train, validation=validation)
        results[name] = {'train_qa': len(train), 'validation_qa': len(validation),
                         'train_image_size': train[0]['image'].size,
                         'validation_image_size': validation[0]['image'].size,
                         'initialization_exists': (root / config['init_from']).exists()}
    report = {'data_binding_status': 'pass', 'training_ready': False, 'configs': results}
    (root / 'refactor_v2/reports/sealed_data_bindings_v3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
