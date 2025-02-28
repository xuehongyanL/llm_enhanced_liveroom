from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from liveroom.llm.batch_inference import (BatchFileClient, BuyStage,
                                          DirectQueryClient, EntryStage,
                                          ReadStage)

parser = argparse.ArgumentParser()

parser.add_argument("--stage", type=str, required=True)
parser.add_argument("--client", type=str, required=True)
parser.add_argument("--df", type=str, required=True)
parser.add_argument("--in_", type=str, required=True)
parser.add_argument("--entry", type=str, required=False)
parser.add_argument("--state", type=str, required=False)
parser.add_argument("--action", type=str, required=False)
parser.add_argument("--read", type=str, required=False)

args = parser.parse_args()

stage_type = args.stage
client_type = args.client

df = pd.read_csv(args.df)
gd = None

if client_type == 'batch':
    client = BatchFileClient(base_url=None, model=None, params={}) # type: ignore
elif client_type == 'direct':
    client = DirectQueryClient(oa=None, model=None, params={}) # type: ignore
else:
    assert False, 'Invalid client.'


if stage_type == 'entry':
    stage = EntryStage(df=df, client=client)
    entry = stage.collect(args.in_)
    np.save(args.entry, entry)
elif stage_type == 'buy':
    stage = BuyStage(df=df, client=client, gd=gd) # type: ignore
    state, action = stage.collect(args.in_)
    np.save(args.state, state)
    np.save(args.action, action)
elif stage_type == 'read':
    stage = ReadStage(df=df, client=client)
    read = stage.collect(args.in_)
    np.save(args.read, read)
else:
    assert False, 'Invalid stage.'
