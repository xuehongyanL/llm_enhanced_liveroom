from typing import List, NamedTuple


class UserCharacter(NamedTuple):
    name: str
    age: str
    consumption: str
    location: str
    others: List[str]
    eng: str


# ‌Z世代‌：指的是年龄在24岁以下的年轻人，主要分布在一二三线城市。
ZShiDai = UserCharacter(name='Z世代',
                        age='18-24岁之间',
                        consumption='消费水平中低',
                        location='一二三线城市',
                        eng='Z',
                        others=[])

# ‌精致妈妈‌：已婚有小孩，消费水平中高以上的女性，年龄在25-35岁之间，主要居住在一二三线城市。
JingZhiMaMa = UserCharacter(name='精致妈妈',
                            age='25-35岁之间',
                            consumption='消费水平中高以上',
                            location='一二三线城市',
                            eng='exquisite_mother',
                            others=['性别为女', '婚育状态为已婚已育'])

# ‌新锐白领‌：未婚（与精致妈妈互斥），消费水平中高，年龄在25-35岁之间，主要居住在一二三线城市。
XinRuiBaiLing = UserCharacter(name='新锐白领',
                              age='25-35岁之间',
                              consumption='消费水平中高',
                              location='一二三线城市',
                              eng='new_white',
                              others=['婚育状态为未婚或未育'])

# ‌资深中产‌：消费水平中高，年龄在36-50岁之间，同样居住在一二三线城市。
ZiShenZhongChan = UserCharacter(name='资深中产',
                                age='36-50岁之间',
                                consumption='消费水平中高',
                                location='一二三线城市',
                                eng='deep_middle',
                                others=[])

# 都市蓝领：年龄在25~35，消费水平属于中低或未显著提升的人群
DuShiLanLing = UserCharacter(name='都市蓝领',
                             age='25-35岁之间',
                             consumption='消费水平中低或未显著提升',
                             location='一二三线城市',
                             eng='city_blue',
                             others=[])

# ‌都市银发‌：年龄在50岁以上，主要居住在一二三线城市。
DuShiYinFa = UserCharacter(name='都市银发',
                           age='50岁以上',
                           consumption='消费水平中等',
                           location='一二三线城市',
                           eng='city_old',
                           others=[])

# ‌小镇青年‌：年龄大致在18-35岁之间，主要生活在四线及以下城市。
XiaoZhenQingNian = UserCharacter(name='小镇青年',
                                 age='18-35岁之间',
                                 consumption='消费水平较低',
                                 location='四线及以下城市',
                                 eng='town_youth',
                                 others=[])

# ‌小镇中老年‌：年龄超过35岁，同样生活在四线及以下城市。
XiaoZhenZhongLaoNian = UserCharacter(name='小镇中老年',
                                     age='35岁以上',
                                     consumption='消费水平较低',
                                     location='四线及以下城市',
                                     eng='town_old',
                                     others=[])


ALL_CHARACTERS = [
    ZShiDai,
    JingZhiMaMa,
    XinRuiBaiLing,
    ZiShenZhongChan,
    DuShiLanLing,
    DuShiYinFa,
    XiaoZhenQingNian,
    XiaoZhenZhongLaoNian,
]
