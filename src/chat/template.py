intent_map = {
    "basic_info": "基本信息",
    "award": "奖励标准",
    "process": "审批流程",
    "materials": "申报材料",
    "condition": "申报条件",
}

PROMPT_TEMPLATE_TOP1 = """【已知】：【"项目名称"：{title}
{label}：{content}】
你现在是一个政企领域的专家，基于【已知】给的多个内容，请回答问题：“{question}”，一定要保证答案正确！"""

PROMPT_TEMPLATE_TOP3 = """【已知】：【"项目名称"：{title1}
{label1}：{content1}】
【"项目名称"：{title2}
{label2}：{content2}】
【"项目名称"：{title3}
{label3}：{content3}】
你现在是一个政企领域的专家，基于【已知】给的多个内容，请回答问题：“{question}”，一定要保证答案正确！"""