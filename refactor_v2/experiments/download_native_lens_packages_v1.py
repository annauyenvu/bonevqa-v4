"""Sequential approved Lens assets using the byte-verifying HF downloader."""
import sys
import download_native_hf_package_v1 as downloader

downloader.PACKAGES.update({
    'smolvlm': dict(
        repo='HuggingFaceTB/SmolVLM-256M-Instruct',
        revision='7e3e67edbbed1bf9888184d9df282b700a323964',
        directory='native_assets_lens_v1/SmolVLM-256M-Instruct',
        files=['README.md', 'added_tokens.json', 'chat_template.json', 'config.json',
               'generation_config.json', 'merges.txt', 'preprocessor_config.json',
               'processor_config.json', 'special_tokens_map.json', 'tokenizer.json',
               'tokenizer_config.json', 'vocab.json', 'model.safetensors']),
    'samhuge': dict(
        repo='facebook/sam-vit-huge',
        revision='87aecf0df4ce6b30cd7de76e87673c49644bdf67',
        directory='native_assets_lens_v1/sam-vit-huge',
        files=['README.md', 'config.json', 'preprocessor_config.json', 'model.safetensors']),
})

if __name__ == '__main__':
    for package in ('smolvlm', 'samhuge'):
        sys.argv = [sys.argv[0], package]
        downloader.main()
