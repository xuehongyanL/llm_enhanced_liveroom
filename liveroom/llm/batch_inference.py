import ast
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Iterator, NamedTuple

import jsonlines
import numpy as np
import openai
from pandas import DataFrame, Series
from tqdm import tqdm

from liveroom.data.character import ALL_CHARACTERS, UserCharacter
from liveroom.data.dataset import GoodsDataset
from liveroom.llm.prompt import (get_buy_prompt, get_entry_prompt,
                                 get_read_prompt, get_sys_prompt)


class RecID(NamedTuple):
    stg: str
    ver: int
    ri: int
    ci: int

    def __repr__(self) -> str:
        return f's{self.stg}_v{self.ver}_i{self.ri}_c{self.ci}'

    @classmethod
    def parse(cls, raw: str) -> 'RecID':
        s, v, i, c = raw.split('_')
        return cls(stg=s[1:], ver=int(v[1:]), ri=int(i[1:]), ci=int(c[1:]))


class Client(ABC):
    @abstractmethod
    def create(self, rec_id: RecID, messages: list[dict]) -> Any:
        pass

    @abstractmethod
    def iter_response(self, in_filename: str) -> Iterator[tuple[RecID, Any]]:
        pass


class Stage(ABC):
    stg: str
    ver: int

    def __init__(self, df: DataFrame, client: Client):
        self.df = df
        self.client = client

    def iter_records(self) -> Iterator[tuple[RecID, Series, UserCharacter]]:
        for row_idx, row in tqdm(self.df.iterrows()):
            for char_idx, char in enumerate(ALL_CHARACTERS):
                rec_id = RecID.parse(f's{self.stg}_v{self.ver}_i{row_idx}_c{char_idx}')
                yield rec_id, row, char

    def batch_generate(self,
                       out_filename: str,
                       partition_fn: Callable[[RecID], bool]): # Process records when partition_fn return True
        with jsonlines.open(out_filename, 'a') as f:
            for rec_id, row, char in self.iter_records():
                if not partition_fn(rec_id):
                    continue
                messages = self.get_messages(row, char)
                output = self.client.create(rec_id, messages)
                f.write(output)

    @abstractmethod
    def get_messages(self, row: Series, char: UserCharacter) -> list[dict]:
        pass

    @abstractmethod
    def collect(self, in_filename: str) -> Any:
        pass


# Output <LLM INPUT> only.
# You should submit <LLM INPUT> manually and retrieve <LLM OUTPUT>.
# Batch inference services are faster and cheaper.
class BatchFileClient(Client):
    def __init__(self, base_url: str, model: str, params: dict):
        self.base_url = base_url
        self.model = model
        self.params = params

    def create(self, rec_id: RecID, messages: list[dict]) -> Any:
        data = dict(custom_id=str(rec_id),
                    method='POST',
                    url=self.base_url,
                    body=dict(model=self.model,
                              messages=messages,
                              response_format={'type': 'json_object'},
                              **self.params))
        return data

    def iter_response(self, in_filename: str) -> Iterator[tuple[RecID, Any]]:
        with open(in_filename) as f:
            for line in f:
                rec = ast.literal_eval(line)
                rec_id = RecID.parse(rec['custom_id'])
                t = rec['response']['body']['choices'][0]['message']['content']
                try:
                    response = ast.literal_eval(t)
                except SyntaxError as e:
                    response = None

                if response is None:
                    try:
                        response = json.loads(t)
                    except json.JSONDecodeError as e:
                        print(t, e)
                        continue
                yield rec_id, response


# Output <LLM OUTPUT> directly.
# Plan-B for open-source LLMs.
class DirectQueryClient(Client):
    def __init__(self, oa: openai.OpenAI, model: str, params: dict):
        self.oa = oa
        self.model = model
        self.params = params

    def create(self, rec_id: RecID, messages: list[dict]) -> Any:
        completion = self.oa.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            response_format={'type': 'json_object'},
            **self.params
        )
        data = completion.model_dump()
        data['req_id'] = str(rec_id)
        return json.dumps(data, indent=None, ensure_ascii=False) + '\n'

    def iter_response(self, in_filename: str) -> Iterator[tuple[RecID, Any]]:
        with open(in_filename) as f:
            for line in f:
                rec = ast.literal_eval(line)
                rec = json.loads(rec)
                rec_id = RecID.parse(rec['req_id'])
                t = rec['choices'][0]['message']['content']
                try:
                    response = ast.literal_eval(t)
                except SyntaxError as e:
                    response = None

                if response is None:
                    try:
                        response = json.loads(t)
                    except json.JSONDecodeError as e:
                        print(t, e)
                        continue
                yield rec_id, response


# Traffic Question
class EntryStage(Stage):
    stg: str = 'entry'
    ver: int = 1

    def get_messages(self, row: Series, char: UserCharacter) -> list[dict]:
        title = row['title']
        dt = datetime.strptime(row['start'], '%Y-%m-%d %H:%M')
        upper_prompt = get_sys_prompt(char, dt)
        lower_prompt = get_entry_prompt(title=title)
        messages = [dict(role='system', content=upper_prompt),
                    dict(role='user', content=lower_prompt)]
        return messages

    def collect(self, in_filename: str) -> np.ndarray:
        entry_mat = np.zeros((len(self.df), len(ALL_CHARACTERS)))
        for rec_id, response in self.client.iter_response(in_filename):
            entry_mat[rec_id.ri][rec_id.ci] = response['want_to_enter']
        return entry_mat


# Interest Question
class BuyStage(Stage):
    stg: str = 'buy'
    ver: int = 1

    def __init__(self, df: DataFrame, client: Client, gd: GoodsDataset, *args, **kwargs):
        super().__init__(df, client, *args, **kwargs)
        self.gd = gd

    def get_messages(self, row: Series, char: UserCharacter) -> list[dict]:
        goods = self.gd[row['start']][:10]
        dt = datetime.strptime(row['start'], '%Y-%m-%d %H:%M')
        upper_prompt = get_sys_prompt(char, dt)
        lower_prompt = get_buy_prompt(goods=goods)
        messages = [dict(role='system', content=upper_prompt),
                    dict(role='user', content=lower_prompt)]
        return messages

    def collect(self, in_filename: str) -> tuple[np.ndarray, np.ndarray]:
        state_mat = np.zeros((len(self.df), len(ALL_CHARACTERS)))
        action_mat = np.zeros((len(self.df), len(ALL_CHARACTERS)))
        for rec_id, response in self.client.iter_response(in_filename):
            if isinstance(response, tuple):
                response = response[0]
            try:
                scores = [res.get('want_to_buy', res.get('want')) for res in response]
            except AttributeError:
                print(response)
                continue
            assert scores[0] is not None
            scores = [v for v in scores if v is not None]
            first_score, avg_score = scores[0], sum(scores) / len(scores)
            state_mat[rec_id.ri][rec_id.ci] = avg_score
            action_mat[rec_id.ri][rec_id.ci] = first_score
        return state_mat, action_mat

# Desire Question
class ReadStage(Stage):
    stg: str = 'read'
    ver: int = 1

    def get_messages(self, row: Series, char: UserCharacter) -> list[dict]:
        dt = datetime.strptime(row['start'], '%Y-%m-%d %H:%M')
        upper_prompt = get_sys_prompt(char, dt)
        lower_prompt = get_read_prompt(top_item_name=row['prod'],
                                       explain=row['text'],
                                       truncate=1024)
        messages = [dict(role='system', content=upper_prompt),
                    dict(role='user', content=lower_prompt)]
        return messages

    def collect(self, in_filename: str) -> np.ndarray:
        read_mat = np.zeros((len(self.df), len(ALL_CHARACTERS)))
        for rec_id, response in self.client.iter_response(in_filename):
            if 'impact' not in response:
                print(response)
                continue
            read_mat[rec_id.ri][rec_id.ci] = response['impact']
        return read_mat
