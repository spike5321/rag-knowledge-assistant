"""生成验收用示例 PDF（一次性脚本，不属于产品代码）。"""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm

pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))

sections = [
    ("新员工入职指南", None),
    ("试用期说明", [
        "新员工试用期为三个月，自入职之日起计算。试用期考核由直属上级在试用期结束前一周发起，"
        "考核结果分为通过、延长观察与不通过三种。延长观察最多一次，时长为一个月。",
        "试用期员工享有全部法定节假日与公司规定的带薪病假，转正后年假按司龄重新核算。",
    ]),
    ("社保与公积金", [
        "公司自入职当月起为员工缴纳五险一金，缴费基数为上一年度月平均工资。公积金缴存比例为个人与"
        "公司各 12%，补充商业医疗保险在转正后的次月生效。",
        "社保基数每年 7 月统一调整一次，调整结果会在当月的工资条中体现。",
    ]),
    ("入职材料清单", [
        "办理入职手续需携带：身份证原件及复印件、学历学位证书原件、离职证明、一寸免冠照片两张，"
        "以及本人名下的一类银行卡复印件。材料齐全后，行政前台会在当天发放工牌与办公设备。",
    ]),
]

c = canvas.Canvas("data/docs/新员工入职指南.pdf", pagesize=A4)
width, height = A4
y = height - 3 * cm
for title, paragraphs in sections:
    if paragraphs is None:
        c.setFont("SimHei", 22)
        c.drawString(2.5 * cm, y, title)
        y -= 1.5 * cm
        continue
    c.setFont("SimHei", 15)
    c.drawString(2.5 * cm, y, title)
    y -= 1.0 * cm
    c.setFont("SimHei", 11)
    for p in paragraphs:
        lines = [p[i:i + 38] for i in range(0, len(p), 38)]
        for line in lines:
            c.drawString(2.5 * cm, y, line)
            y -= 0.7 * cm
    y -= 0.6 * cm
    if y < 3 * cm:
        c.showPage()
        y = height - 3 * cm
c.save()
print("PDF 已生成")
