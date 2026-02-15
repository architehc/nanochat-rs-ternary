# Deployment

Production flow:

1. Export checkpoint to GGUF + mHC
2. Start `nanochat-serve`
3. Monitor `/health` and `/metrics`
4. Track latency, tokens/sec, and error-rate
