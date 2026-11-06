"""Read-only local cache inventory, no downloads or checkpoint deserialization."""
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUBS = [ROOT / '.evaluation_runtime/hf_cache/hub', Path('/Users/uyen/.cache/huggingface/hub')]


def main():
    roots = {}
    for hub in HUBS:
        entries = {}
        if hub.exists():
            for model in sorted(hub.glob('models--*')):
                snapshots = []
                for revision in sorted((model / 'snapshots').glob('*')):
                    files = []
                    for path in sorted(revision.rglob('*')):
                        if path.is_file() or path.is_symlink():
                            exists = path.exists()
                            files.append({'relative_path': str(path.relative_to(revision)), 'exists': exists, 'bytes': path.stat().st_size if exists else None})
                    snapshots.append({'revision': revision.name, 'files': files})
                entries[model.name] = snapshots
        roots[str(hub)] = entries
    models = sorted({name for entries in roots.values() for name in entries})
    matches = {name: [m for m in models if any(token in m.lower() for token in terms)] for name, terms in {
        'RoBERTa': ['roberta'], 'M3AE': ['m3ae'], 'Vicuna': ['vicuna'],
        'SmolVLM': ['smolvlm'], 'EVA_CLIP': ['eva-clip', 'eva_clip'],
        'SAM_Med2D': ['sam-med2d', 'sam_med2d'], 'SAM_vit_huge': ['sam-vit-huge'],
        'MedSAM_distinct_model': ['medsam'],
    }.items()}
    versions = {}
    for package in ('torch', 'transformers', 'peft', 'timm', 'pytorch-lightning'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        'scope': 'Two explicitly listed HuggingFace cache roots only; not a whole-machine or standalone-checkpoint inventory. Filename matches are not architecture/weight/provenance verification. .no_exist markers excluded.',
        'roots': roots, 'model_id_matches': matches, 'runtime_versions': versions,
        'no_network': True, 'no_weight_loading': True, 'no_installation': True,
        'native_reference_released': False,
    }
    output = ROOT / 'refactor_v2/reports/paper_asset_inventory_v1'
    output.mkdir(exist_ok=False)
    with (output / 'cache_inventory.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'models_found': models, 'matches': matches, 'runtime_versions': versions}, indent=2))


if __name__ == '__main__':
    main()
