# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/enoonc-sketch/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/enoonc-sketch/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs (OpenAI Chat Completions and Anthropic Messages, JSON and streaming), over ten free-tier providers. It reads the trial brief, probes every provider's catalog route, registers its port and catalog with the environment, and answers each of the 20 requests with the best free model that meets the request's requirements — tools, vision, JSON mode, or long context — retrying the same model once when a provider refuses (402/429/5xx), then failing over to the next best model. A provider that just failed is skipped for the next request; one that fails three times in a row is treated as spent while any other candidate remains. JSON-mode requests on the OpenAI wire never go to an Anthropic-wire provider, and streaming always carries usage (OpenAI `stream_options.include_usage`, two-pass usage collection for Anthropic streams). Python 3.13 standard library only, one file, no dependencies to install, deterministic: the same trial replays to the same state export. Local practice trials 0, 1 and 2 each score 99 (20/20 valid).

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:18080 MAKEFUN_PROMPT_FILE=brief.txt python3 main.py
```

---

Entered on [make.fun](https://make.fun/?ref=cs36dnd5&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
