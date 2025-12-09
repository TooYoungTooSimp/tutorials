from gpt_common import *
from glob import glob

sysprompt = "你是一名中石化新入职大学生，正在听领导的讲座并写感想。讲座材料以听写转录的形式给出，注意同音字近音字的存在，从事实（O），感受（R），理解（I），行动（D）四个方面每个方面不超过100字写出感想"

txts = glob("*.txt")
for idx, txt in enumerate(txts):
    print(f"{idx}: {txt}")
idx = input(":> ")
txt = txts[int(idx)]
target_filename = txt.replace(".txt", "-ORID.txt")
with open(txt, "r", encoding="utf-8") as f:
    txt = f.read()
hist: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": sysprompt,
    },
    {
        "role": "user",
        "content": txt,
    },
]
if __name__ == "__main__":
    # res = ask_gpt(hist, model="gpt-4.5-preview")
    res = ask_gpt(hist, model="gpt-4.1")
    with open(target_filename, "w", encoding="utf-8") as f:
        f.write(res)
