# DD3412-chain-of-verification-reproduction

To train CoVe, run:

```bash
    python -m main -m llama-65b -t wikidata_category -s two_step -temp 0.07 -p 0.9
```

- Available models are: `llama2`, `llama2_70b`, `llama-65b`, `gpt3`
- Available settings are: `joint`, `two_step`, `factored`
- Available tasks are: `wikidata`, `wikidata_category`, `multispanqa` 