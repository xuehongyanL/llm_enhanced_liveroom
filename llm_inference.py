import argparse
import os

import pandas as pd
from openai import OpenAI

from liveroom.data.dataset import GoodsDataset
from liveroom.llm.batch_inference import (BatchFileClient, BuyStage,
                                          DirectQueryClient, EntryStage,
                                          ReadStage, RecID)

parser = argparse.ArgumentParser()

parser.add_argument("--stage", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--client", type=str, required=True)
parser.add_argument("--df", type=str, required=True)
parser.add_argument("--gd", type=str, required=False)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--split", type=str, required=True)

args = parser.parse_args()

stage_type = args.stage
model = args.model
client_type = args.client

base_url = os.environ['OPENAI_BASE_URL']
api_key = os.environ['OPENAI_API_KEY']

params = dict(temperature=0.2, max_tokens=1000)

df = pd.read_csv(args.df)
if stage_type == 'buy':
    gd = GoodsDataset(args.gd)

if client_type == 'batch':
    client = BatchFileClient(base_url=base_url, model=model, params=params)
elif client_type == 'direct':
    oa = OpenAI(base_url=base_url, api_key=api_key)
    client = DirectQueryClient(oa=oa, model=model, params=params)
else:
    assert False, 'Invalid client.'


if stage_type == 'entry':
    stage = EntryStage(df=df, client=client)
elif stage_type == 'buy':
    stage = BuyStage(df=df, client=client, gd=gd)
elif stage_type == 'read':
    stage = ReadStage(df=df, client=client)
else:
    assert False, 'Invalid stage.'


# e.g. 0/8
l, r = args.split.split('/')
l, r = int(l), int(r)


# TODO: You can customize this fn creatively, e.g. skipping existing records.
def partition_fn(rec_id: RecID) -> bool:
    return rec_id.ri % r == l


stage.batch_generate(out_filename=args.out, partition_fn=partition_fn)
