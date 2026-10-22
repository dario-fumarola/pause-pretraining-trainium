# Provenance

Started October 22, 2026 from the owner-directed pause pilot prepared locally in useful-thoughts-trainium, predecessor commit 84b165f plus the documented whole-instance adaptation. No predecessor Git history or large artifacts are copied into this new repository.

The Qwen causal cache/trunk methods, masked causal CE, process-local runtime compatibility, BF16 parameter casting, FP32 optimizer digests, immutable artifact helpers and sampled fingerprints are retained from the predecessor. Unused policy/reader interfaces were omitted; their numerical code was not changed. The focused CPU tests compare the resulting path with independent standard Hugging Face forwards and gradients.

The architecture config is the official Qwen/Qwen3-0.6B-Base config, SHA256 504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59. The existing tokenizer JSON exactly matches the official 0.6B tokenizer, SHA256 c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539. No pretrained weights are used.

The original 1.7B result report and numerical sources remain at https://github.com/dario-fumarola/useful-thoughts-trainium. The local predecessor preservation receipt checks 2,516 original tracked numerical/config/evidence files against starting commit d07d89e without changes. This is not a new full-payload rehash of old native checkpoints.
