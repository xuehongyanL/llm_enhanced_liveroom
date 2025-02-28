from __future__ import annotations

import json
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class JSONDataset(dict, Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            raw = json.load(f)
        super().__init__(raw)

    def __repr__(self) -> str:
        return Dataset.__repr__(self) # Prevent printing full dict


class Goods(NamedTuple):
    idx: int
    name: str
    price: float


class GoodsDataset(JSONDataset):
    def __getitem__(self, timeline: datetime | str) -> list[Goods]:
        if not isinstance(timeline, str):
            timeline = timeline.strftime('%Y-%m-%d %H:%M')
        if (items := self.get(timeline)) is not None:
            return [Goods(*item) for item in items if item[-1] > 0]
        else:
            raise KeyError(timeline)


class ExplainDataset(JSONDataset):
    def __getitem__(self, timeline: datetime | str) -> str | None:
        if not isinstance(timeline, str):
            timeline = timeline.strftime('%Y-%m-%d %H:%M')
        return self.get(timeline, None)


class FusionDataset(Dataset):
    def __init__(self, filename: str, tokenizer: Any = None):
        super().__init__()
        self.df = pd.read_csv(filename)

        self.tokenizer = tokenizer

        self.text_columns = ['title',
                             'prod',
                             'text',]
        self.x_columns = ['dow_cos',
                          'dow_sin',
                          'hour_cos',
                          'hour_sin',
                          'is_workday',
                          'is_holiday',
                          'duration',
                          'user_cnt',]
        self.y_columns = ['log_gmv',
                          'log_deal_order_cnt',
                          'log_live_deal_user_cnt',
                          'log_watch_user_cnt',
                          'prop_Z_watch',
                          'prop_exquisite_mother_watch',
                          'prop_new_white_watch',
                          'prop_deep_middle_watch',
                          'prop_city_blue_watch',
                          'prop_city_old_watch',
                          'prop_town_youth_watch',
                          'prop_town_old_watch',
                          'prop_Z_pay',
                          'prop_exquisite_mother_pay',
                          'prop_new_white_pay',
                          'prop_deep_middle_pay',
                          'prop_city_blue_pay',
                          'prop_city_old_pay',
                          'prop_town_youth_pay',
                          'prop_town_old_pay',]

    def __len__(self):
        return len(self.df)

    def tokenize(self, title, prod, text, max_length=512):
        t = '[CLS]' + title + '[SEP]' + prod + '[SEP]' + text + '[SEP]'
        tokens = self.tokenizer.tokenize(t)

        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + ['[SEP]']

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        return ids

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        title, prod, text = row[self.text_columns]
        t = np.array(self.tokenize(title, prod, text))
        x = row[self.x_columns].values.astype(np.float32)
        y = row[self.y_columns].values.astype(np.float32)

        return dict(text=t, x=x, y=y)

class BertDataset(FusionDataset):
    def __init__(self, filename: str, features):
        super().__init__(filename, None)
        self.features = np.load(features)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        title, prod, text = row[self.text_columns]
        t = self.features[idx].astype(np.float32)
        x = row[self.x_columns].values.astype(np.float32)
        y = row[self.y_columns].values.astype(np.float32)

        return dict(text=t, x=x, y=y)

class LLMEnhancedDataset(Dataset):
    def __init__(self,
                 fusion: FusionDataset,
                 entry: str,
                 state: str,
                 action: str,
                 read: str):
        super().__init__()
        self.fusion = fusion

        self.entry = np.load(entry)
        self.state = np.load(state)
        self.action = np.load(action)
        self.read = np.load(read)

        self.llm_x_features = ['pred_Z_entry',
                             'pred_exquisite_mother_entry',
                             'pred_new_white_entry',
                             'pred_deep_middle_entry',
                             'pred_city_blue_entry',
                             'pred_city_old_entry',
                             'pred_town_youth_entry',
                             'pred_town_old_entry',
                             'pred_Z_action',
                             'pred_exquisite_mother_action',
                             'pred_new_white_action',
                             'pred_deep_middle_action',
                             'pred_city_blue_action',
                             'pred_city_old_action',
                             'pred_town_youth_action',
                             'pred_town_old_action',
                             'pred_Z_read',
                             'pred_exquisite_mother_read',
                             'pred_new_white_read',
                             'pred_deep_middle_read',
                             'pred_city_blue_read',
                             'pred_city_old_read',
                             'pred_town_youth_read',
                             'pred_town_old_read',
                             'pred_Z_state',
                             'pred_exquisite_mother_state',
                             'pred_new_white_state',
                             'pred_deep_middle_state',
                             'pred_city_blue_state',
                             'pred_city_old_state',
                             'pred_town_youth_state',
                             'pred_town_old_state',]

    def __len__(self):
        return len(self.fusion)

    def __getitem__(self, idx):
        row = self.fusion.df.iloc[idx]
        x = row[self.fusion.x_columns].values.astype(np.float32)
        y = row[self.fusion.y_columns].values.astype(np.float32)

        entry = self.entry[idx].astype(np.float32)
        action = self.action[idx].astype(np.float32)
        read = self.read[idx].astype(np.float32)
        state = self.state[idx].astype(np.float32)
        x = np.concatenate((x, entry, action, read, state))
        w = (action - state).mean()

        return dict(x=x, y=y, w=w)
