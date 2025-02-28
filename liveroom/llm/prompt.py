from datetime import datetime, timedelta
from string import Template

from chinese_calendar import get_holiday_detail

from liveroom.data.character import UserCharacter
from liveroom.data.dataset import Goods

holiday_translate = {
    "New Year's Day": "元旦",
    "Spring Festival": "春节",
    "Tomb-sweeping Day": "清明节",
    "Labour Day": "劳动节",
    "Dragon Boat Festival": "端午节",
    "National Day": "国庆节",
    "Mid-autumn Festival": "中秋节",
}


def describe_date(dt: datetime):
    date = datetime.strftime(dt, '%Y年%m月%d日')
    weekday = '一二三四五六日'[dt.weekday()]
    today_is_holiday, holiday = get_holiday_detail(dt)
    ret = f'{date}星期{weekday}，'
    if today_is_holiday:
        if holiday is not None:
            ret += f'处于{holiday_translate[holiday]}假期，是休息日'
        else:
            ret += '是休息日'
    else:
        if holiday is not None:
            ret += f'但因为{holiday_translate[holiday]}调休，是工作日'
        else:
            ret += '是工作日'
    return ret


def describe_datetime(dt: datetime):
    curr_time = datetime.strftime(dt, '%H:%M')
    if dt.hour <= 5:
        return f'现在的时间是凌晨{curr_time}，接下来的白天是{describe_date(dt)}。'
    elif dt.hour >= 21:
        return f'现在的时间是晚上{curr_time}，今天是{describe_date(dt)}，明天是{describe_date(dt + timedelta(days=1))}。'
    else:
        return f'现在的时间是{curr_time}，今天是{describe_date(dt)}。'


def describe_goods(goods: list[Goods]):
    return ''.join(f'#{i}【{good.name}】单价为【{good.price:.2f}】元\n' for i, good in enumerate(goods, start=1))


def describe_character(char: UserCharacter):
    if char.others:
        others = f'此外你还有以下特征：{"、".join(char.others)}。'
    else:
        others = ''

    return f'''
你的用户身份被定义为【{char.name}】;
你所处的年龄段为：{char.age}；
你生活的城市类别为：{char.location}；
你的消费水平为：{char.consumption}；
{others}
        '''


def get_sys_prompt(character: UserCharacter, dt: datetime) -> str:
    sys_prompt = Template('''你擅长角色扮演。接下来你将扮演一个用户，尝试访问一个带货的手机直播间。
你扮演的角色身份有以下特点：
${character}
${datetime}
接下来你将阅读文本格式的直播内容和商品信息等，并按照我的要求，结合你的身份做出决策。
''')
    return sys_prompt.substitute(character=describe_character(character),
                                 datetime=describe_datetime(dt))


def get_entry_prompt(title: str) -> str:
    entry_example = [
        {"want_to_enter": 1, "reason": "1. 作为学生，我没有足够的钱购买首饰。2. 现在是工作日的上课时间，我不太可能使用手机。"},
        {"want_to_enter": 5, "reason": "1. 作为新锐白领，我对化妆品比较感兴趣。2. 现在是凌晨，白天是工作日，我现在更想休息。"},
    ]

    user_prompt = Template(f'''## 任务说明
    现在你看到直播间的标题是【{"${title}"}】。
    你可以假设主播是可信且受欢迎的，这一点不需要作为你决策的依据。
    请结合身份、兴趣爱好、消费能力、作息时间、是否空闲等因素，评估你是否可能且有意愿进入这个直播间。
    首先将你的意愿用一个1到5之间的整数表示：1表示几乎不可能或没有意愿，5表示可能且有极大的意愿。
    接着为你的打分，给出一个简短清晰的理由。

    ##输出格式
    您应该始终遵循指令并输出一个有效的JSON对象。以下是示例，仅供你参考格式：
    {entry_example[0]}
    {entry_example[1]}
    ''')

    return user_prompt.substitute(title=title)


def get_buy_prompt(goods) -> str:
    buy_example = [
        {"idx": 1, "want_to_buy": 1, "reason": "1. 作为学生，我对首饰不感兴趣。2. 商品单价过高，我很难负担得起。"},
        {"idx": 2, "want_to_buy": 3, "reason": "1. 作为都市蓝领，我对这双鞋感兴趣，有购买意愿。2. 我对价格比较敏感，需要进一步观看直播才能作出决定。"},
        {"idx": 3, "want_to_buy": 5, "reason": "1. 作为资深中产，我正需要为过年采购白酒作为礼品。2. 我对价格不太敏感，相信价格已经足够实惠。"},
    ]

    buy_prompt = Template(f'''## 任务说明
    现在假设你最终进入了直播间（即使你认为没有意愿这么做）。
    点开右下角的“商品橱窗”，你会看到以下商品：
    {"${goods}"}
    请结合身份、兴趣爱好、消费能力等因素，评估你是否有以下的意愿购买这些商品。
    对橱窗中的每个商品依次进行如下的动作：
    首先将你的意愿分别用一个1到5之间的整数表示，1表示没有意愿，5表示有极大的意愿。
    接着为你的打分，给出一个简短清晰的理由。

    ##输出格式
    您应该始终遵循指令并输出一个有效的JSON列表对象。以下是一些示例，仅供你参考格式。
    请注意，示例仅供参考，请遵守你自己的身份。
    {buy_example}
    ''')

    return buy_prompt.substitute(goods=describe_goods(goods))


def get_read_prompt(top_item_name: str, explain: str, truncate=1024) -> str:
    read_example = [
        {"impact": 2, "reason": "1. 作为学生，我对价格比较敏感，主播并没有体现出商品的价格优势。"},
        {"impact": 3, "reason": "1. 我本来就很想购买这件商品，讲解的内容对此影响不大。"},
        {"impact": 5, "reason": "1. 主播提到了即将开始“秒杀”活动，届时价格将会非常实惠。"},
    ]

    read_prompt = Template(f'''## 任务说明
    现在假设你尝试观看直播的内容。
    可以看到直播间现在正在讲解的商品是【{"${top_item_name}"}】。
    你看到的讲解内容转换为文字如下：
    【讲解内容开始】
    {"${explain}"}
    【讲解内容结束】
    由于技术原因，文本中可能存在很多语气词和转译错误，请你尽可能地结合上下文理解其含义。
    请结合身份、兴趣爱好、消费能力等因素，评估这些讲解对你购买该商品的影响。
    首先将你的意愿分别用一个1到5之间的整数表示，1表示有强烈的负面影响，3表示没有影响，5表示有强烈的正面影响。
    接着为你的打分，给出一个简短清晰的理由。

    ##输出格式
    您应该始终遵循指令并输出一个有效的JSON对象。以下是一些示例，仅供你参考格式。
    {read_example[0]}
    {read_example[1]}
    {read_example[2]}
    ''')

    if truncate > 0:
        explain = explain[:truncate]

    return read_prompt.substitute(top_item_name=top_item_name,
                                  explain=explain)