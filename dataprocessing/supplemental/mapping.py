"""CoxPWD v3 和 Indoor Waste v1 的来源类别及逐物体审核规则。"""

COXPWD_RULES = {
    'bottle': ('target', 2, '可辨认的瓶身，瓶盖可有可无'),
    'cup': ('target', 4, '可辨认的杯身，材质不限'),
    'bottle_cap': ('background', None, '独立瓶盖不计瓶'),
    'straw': ('background', None, '独立吸管不计包装物或杯'),
    'spoon': ('background', None, '独立勺不计杯'),
    'rope': ('background', None, '独立绳不计包装物'),
    'net': ('background', None, '网具不计包装物'),
    'sunglass': ('background', None, '眼镜不属于五类目标'),
    'toy': ('background', None, '玩具不属于五类目标'),
    'fishing_item': ('review', None, '渔具名下物体须复核形状和用途'),
    'bag': ('review', None, '确认是柔性包装袋才计包装物'),
    'polythene': ('review', None, '确认是柔性包装膜才计包装物，无法辨认用途时留出'),
    'packet': ('review', None, '确认是柔性包装才计包装物，硬质容器按形态判定'),
    'others': ('review', None, '其他物体须根据五类定义逐个判定'),
}

HARD_BACKGROUNDS = frozenset({'bottle_cap', 'straw', 'spoon', 'rope'})


INDOOR_RULES = {
    'cap': ('background', None, '独立瓶盖作为 bottle_cap 背景，不计瓶'),
    'shell': ('background', None, '果壳、种壳等独立硬壳作为 shell 背景，不属于五类目标'),
    'paper': ('review', None, '纸片计纸类，包装纸计包装物，纸杯计杯；逐件复核'),
    'plastic': ('review', None, '确认形态和用途后逐件归类；明确只是塑料碎片时可标背景，难辨时排除'),
}

INDOOR_BACKGROUNDS = frozenset({'bottle_cap', 'shell', 'plastic_fragment'})
