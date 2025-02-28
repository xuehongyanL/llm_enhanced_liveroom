# llm_enhanced_liveroom

## Data

We sampled 10 pitches from our private dataset. There are two  datasets:

1. Basic data, including pitch's time, title, speech, and sales returns such as GMV and view count.

2. Showcase data, including list of products corresponding to each pitch, including the index, name, and price of each item.

NOTICE: At the request of the data provider, all brand names and person names in the text data have been anonymized, and all numerical data is randomly generated.

## Reproduction

### Overview

As shown in our paper, there are two steps of interaction: 1. LLM-enhanced feature scoring, and 2. Sales prediction. Note that causal debiasing is just a plugin in out framework.

### Query LLMs for features

In this step, we perform Traffic Prediction, Interest Prediction, and Desire Prediction. In our code, we define them as three STAGES, called "entry", "buy", and "read" correspondingly.

Specially, in "buy" stage, we pick two values from LLM: the average score and a score of the first item (the promoting item), obtaining 16-dimensional scores. These scores are called "state" and "action" correspondingly, somehow following the RL notations.

In general, we perform the three stages of "entry", "buy", and "read", and obtain the four collections of scores: "entry", "state", "action", and "read".

You can query LLMs in two modes:

#### Batch mode

Some closed-source services support OpenAI-compatible batch API, e.g. Qwen-Plus and GLM-4-Air. In this mode, you can generate a large number of API requests as one file.

1. Generate requests (GLM-4-Flash as example)

```shell
cd <CODE_BASE>
export OPENAI_BASE_URL='/v4/chat/completions'

python llm_inference.py --stage=entry --model=glm-4-flash --client=batch --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/entry_in_sample.jsonl --split=0/1
python llm_inference.py --stage=buy --model=glm-4-flash --client=batch --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/buy_in_sample.jsonl --split=0/1
python llm_inference.py --stage=read --model=glm-4-flash --client=batch --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/read_in_sample.jsonl --split=0/1
```

2. Submit requests

You can upload `./data/*_in_sample.jsonl` [manually or via API](https://www.bigmodel.cn/dev/howuse/batchapi).

When your batches are completed, download them, then rename them as `./data/*_out_sample.jsonl`

#### Direct mode

When batch API is unavailable, you can perform conventional OpenAI-compatible requests.

1. Send requests (GLM-4-Flash as example)

```shell
export OPENAI_BASE_URL='https://open.bigmodel.cn/api/paas/v4'
export OPENAI_API_KEY='<YOUR_KEY_HERE>'
python llm_inference.py --stage=entry --model=glm-4-flash --client=direct --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/entry_out_sample.jsonl --split=0/1
python llm_inference.py --stage=buy --model=glm-4-flash --client=direct --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/buy_out_sample.jsonl --split=0/1
python llm_inference.py --stage=read --model=glm-4-flash --client=direct --df=./data/pitch_sample.csv --gd=./data/goods_sample.json --out=./data/read_out_sample.jsonl --split=0/1
```

Note that this method may be very slow (especially at "buy" stage). You can run multiple processes to speed up. For example, you can run 8 processes, setting from `--split=0/8` to `--split=7/8`. When you set `i/8`, any row satisfying `row_idx % 8 != i` will be skipped in this process. Finally you can manually concat all parts together.

### Collect LLM features

```shell
python collect_llm_features.py --stage=entry --client=<batch/direct> --df=./data/pitch_sample.csv --in_=./data/entry_out_sample.jsonl --entry=./data/entry_sample.npy
python collect_llm_features.py --stage=buy --client=<batch/direct> --df=./data/pitch_sample.csv --in_=./data/buy_out_sample.jsonl --state=./data/state_sample.npy --action=./data/action_sample.npy
python collect_llm_features.py --stage=read --client=<batch/direct> --df=./data/pitch_sample.csv --in_=./data/read_out_sample.jsonl --read=./data/read_sample.npy
```

### End-to-end Sales prediction

```shell
python main_exp.py --task=causal --seed=<SEED> --device=<cuda:X> --df=./data/pitch_sample.csv --entry=./data/entry_sample.npy --state=./data/state_sample.npy --action=./data/action_sample.npy --read=./data/read_sample.npy --out_dir=./out --n_epoch=10
```

Checkpoints are saved in `./out/`