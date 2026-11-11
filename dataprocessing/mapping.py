"""按 TACO 原始类别名称映射五类目标；形态有歧义的类别必须逐对象复核。"""

from typing import Any


PROJECT_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "罐"},
    {"id": 2, "name": "瓶"},
    {"id": 3, "name": "包装物"},
    {"id": 4, "name": "杯"},
    {"id": 5, "name": "纸类物品"},
]

# 规则按名称而非原始编号匹配。包装物限柔性包装袋、膜及包装纸；
# 金属、塑料或纸材料本身都不能使硬质容器成为目标包装物。
# 每项依次为：状态、唯一项目类别编号、中文依据。
_RULES: dict[str, tuple[str, int | None, str]] = {
    "Aluminium foil": (
        "target", 3,
        "铝箔属于柔性薄膜包装，按包装形态归包装物，不因金属材料归罐。",
    ),
    "Battery": (
        "background", None,
        "电池是独立电气物品，不属于罐、瓶、柔性包装、一次性杯或纸片。",
    ),
    "Aluminium blister pack": (
        "background", None,
        "铝制泡罩是成型泡罩包装，不是柔性包装袋或膜，不按包装用途泛化为目标。",
    ),
    "Carded blister pack": (
        "background", None,
        "卡纸泡罩是复合成型包装，不等同纸片或柔性包装袋、膜。",
    ),
    "Other plastic bottle": (
        "target", 2,
        "类别明确为塑料瓶，瓶体形态确定，不以透明度或塑料种类排除。",
    ),
    "Clear plastic bottle": (
        "target", 2,
        "类别明确为透明塑料瓶，按瓶体形态归瓶。",
    ),
    "Glass bottle": (
        "target", 2,
        "类别明确为玻璃瓶，按瓶体形态归瓶，不将所有玻璃物归瓶。",
    ),
    "Plastic bottle cap": (
        "background", None,
        "塑料瓶盖是独立配件而非瓶体，不因来源于瓶而计为瓶。",
    ),
    "Metal bottle cap": (
        "background", None,
        "金属瓶盖是独立配件而非瓶体或罐体。",
    ),
    "Broken glass": (
        "review", None,
        "碎玻璃可能来自目标玻璃瓶，也可能来自杯或其他玻璃物，须逐对象确认来源和形态。",
    ),
    "Food Can": (
        "target", 1,
        "食品金属罐具有明确罐体形态，归罐。",
    ),
    "Aerosol": (
        "background", None,
        "气雾剂压力容器不属于本项目限定的饮料罐或食品罐。",
    ),
    "Drink can": (
        "target", 1,
        "饮料金属罐具有明确罐体形态，归罐。",
    ),
    "Toilet tube": (
        "review", None,
        "卫生纸芯是立体纸筒，不能仅因纸材料自动当作纸片，需复核项目边界。",
    ),
    "Other carton": (
        "review", None,
        "其他纸盒可能完整或压扁，类别不能证明为纸板片，需逐对象复核。",
    ),
    "Egg carton": (
        "review", None,
        "蛋盒是成型纸质容器，不自动当作纸板片或柔性包装，需复核实际形态。",
    ),
    "Drink carton": (
        "review", None,
        "饮料纸盒是立体复合容器，不是瓶，不能自动当作纸板片，需复核实际形态。",
    ),
    "Corrugated carton": (
        "review", None,
        "瓦楞纸箱可能完整或展开，类别不等于纸板片，需逐对象复核。",
    ),
    "Meal carton": (
        "review", None,
        "餐盒是成型纸质容器，不能自动当作纸片或柔性包装，需复核实际形态。",
    ),
    "Pizza box": (
        "review", None,
        "披萨盒属于纸箱形态，不能自动当作纸板片，需逐对象复核。",
    ),
    "Paper cup": (
        "target", 4,
        "纸杯属于明确一次性杯形态，按杯归类，不因纸材料重复归纸类物品。",
    ),
    "Disposable plastic cup": (
        "target", 4,
        "类别明确为一次性塑料杯，归杯。",
    ),
    "Foam cup": (
        "target", 4,
        "泡沫杯属于明确一次性杯形态，归杯而非其他泡沫包装。",
    ),
    "Glass cup": (
        "review", None,
        "玻璃杯不是明确一次性杯，不能直接纳入目标杯，需人工复核项目边界。",
    ),
    "Other plastic cup": (
        "review", None,
        "其他塑料杯未明确一次性或可重复使用，需逐对象复核。",
    ),
    "Food waste": (
        "background", None,
        "食物残渣不是五类目标物体，不能按邻近包装推断其类别。",
    ),
    "Glass jar": (
        "background", None,
        "玻璃广口罐属于硬质容器，不是目标金属罐或瓶，也不是柔性包装。",
    ),
    "Plastic lid": (
        "background", None,
        "塑料盖是独立配件，不是杯体、瓶体或柔性包装。",
    ),
    "Metal lid": (
        "background", None,
        "金属盖是独立配件，不是目标罐体或瓶体。",
    ),
    "Other plastic": (
        "review", None,
        "其他塑料无法确定物体形态，可能含瓶、杯或包装碎片，须逐对象复核。",
    ),
    "Magazine paper": (
        "target", 5,
        "杂志纸属于明确纸张形态，归纸类物品。",
    ),
    "Tissues": (
        "target", 5,
        "纸巾属于明确纸张形态，归纸类物品。",
    ),
    "Wrapping paper": (
        "target", 3,
        "包装纸按柔性包裹形态唯一归包装物，不因纸材料重复归纸类物品。",
    ),
    "Normal paper": (
        "target", 5,
        "普通纸属于明确纸张形态，归纸类物品。",
    ),
    "Paper bag": (
        "target", 3,
        "纸袋按柔性包装袋形态唯一归包装物，不因纸材料重复归纸类物品。",
    ),
    "Plastified paper bag": (
        "target", 3,
        "覆塑纸袋仍为柔性包装袋，按形态唯一归包装物。",
    ),
    "Plastic film": (
        "target", 3,
        "塑料薄膜属于柔性包装膜，归包装物而非硬质塑料容器。",
    ),
    "Six pack rings": (
        "background", None,
        "六联提环是环状连接件，不是包装袋或膜，不能仅因包装用途纳入目标。",
    ),
    "Garbage bag": (
        "target", 3,
        "垃圾袋属于柔性塑料袋，按袋状形态归包装物。",
    ),
    "Other plastic wrapper": (
        "target", 3,
        "塑料包裹物属于柔性包装膜，归包装物，不扩展到其他硬质塑料。",
    ),
    "Single-use carrier bag": (
        "target", 3,
        "一次性购物袋属于柔性包装袋，归包装物。",
    ),
    "Polypropylene bag": (
        "target", 3,
        "聚丙烯袋属于柔性包装袋，按袋状形态归包装物。",
    ),
    "Crisp packet": (
        "target", 3,
        "薯片袋属于柔性食品包装袋，归包装物。",
    ),
    "Spread tub": (
        "background", None,
        "涂抹食品盒是硬质塑料容器，不是目标罐、杯或柔性包装袋、膜。",
    ),
    "Tupperware": (
        "background", None,
        "保鲜盒是可重复使用的硬质容器，不属于柔性包装或目标一次性杯。",
    ),
    "Disposable food container": (
        "background", None,
        "一次性餐盒仍是成型容器，不因一次性或包装用途归柔性包装或杯。",
    ),
    "Foam food container": (
        "background", None,
        "泡沫餐盒是成型容器，不是泡沫杯，也不是柔性包装袋、膜。",
    ),
    "Other plastic container": (
        "background", None,
        "其他塑料容器属于容器形态，不把硬质包装无限扩展为目标包装物。",
    ),
    "Plastic glooves": (
        "background", None,
        "塑料手套是穿戴用品，即使柔软也不是包装袋或膜。",
    ),
    "Plastic utensils": (
        "background", None,
        "塑料餐具是器具，不是目标杯或柔性包装。",
    ),
    "Pop tab": (
        "background", None,
        "拉环是罐的独立配件，不是罐体，不按来源计为罐。",
    ),
    "Rope & strings": (
        "background", None,
        "绳线是条状物，不是柔性包装袋、膜或其他目标形态。",
    ),
    "Scrap metal": (
        "review", None,
        "废金属可能含食品罐或饮料罐碎片，类别不能确定来源，须逐对象复核。",
    ),
    "Shoe": (
        "background", None,
        "鞋是穿戴用品，不属于五类目标。",
    ),
    "Squeezable tube": (
        "background", None,
        "挤压软管是带管口的管状容器，不是瓶或柔性包装袋、膜。",
    ),
    "Plastic straw": (
        "background", None,
        "塑料吸管是细管状器具，不属于杯体或柔性包装。",
    ),
    "Paper straw": (
        "background", None,
        "纸吸管是管状器具，不能仅因纸材料归纸片。",
    ),
    "Styrofoam piece": (
        "review", None,
        "泡沫碎片可能来自目标泡沫杯或非目标缓冲材料、餐盒，须逐对象复核。",
    ),
    "Unlabeled litter": (
        "review", None,
        "未标明类别的垃圾可能包含目标或其碎片，无法仅凭原类别作可靠映射。",
    ),
    "Cigarette": (
        "background", None,
        "烟蒂是烟草制品残留，不因含纸材料归纸类物品。",
    ),
}


def build_mapping(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成按输入顺序排列的、每个原始类别最多对应一个目标的映射表。

    参数：
        categories（list[dict[str, Any]]）：原始类别字典列表，每项必须含
            id（整数类别标识）、name（英文名称字符串）、supercategory
            （原始上级类别字符串）；无物理单位或坐标。按名称精确匹配规则，
            不假设原始类别编号；不修改输入，也不读取或写入文件。

    返回：
        list[dict[str, Any]]：每个输入类别对应一项，保留
        original_category_id、original_name、supercategory，并提供
        status（target、background 或 review）、project_category_id
        （整数或 None）、project_category_name（中文字符串或 None）和
        reason（中文依据）。未知名称一律为 review，项目类别字段为 None。
        复核状态不表示已完成任何人工确认。

    异常：
        KeyError：类别缺少 id、name 或 supercategory 必需字段。
        TypeError：类别条目不是可按字符串键访问的字典，或 name 不可哈希。
    """
    project_names = {category["id"]: category["name"] for category in PROJECT_CATEGORIES}
    result: list[dict[str, Any]] = []
    for category in categories:
        name = category["name"]
        status, project_id, reason = _RULES.get(
            name,
            (
                "review",
                None,
                "未识别的新类别，没有经过确认的形态和材料边界规则，必须人工复核。",
            ),
        )
        result.append(
            {
                "original_category_id": category["id"],
                "original_name": name,
                "supercategory": category["supercategory"],
                "status": status,
                "project_category_id": project_id,
                "project_category_name": project_names.get(project_id),
                "reason": reason,
            }
        )
    return result
