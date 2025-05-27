import pandas as pd

begin = r"\begin{thebibliography}{10}"
end = r"\end{thebibliography}"

with open("../論文排版/format/ref.tex", encoding="utf-8") as fp:
    s = fp.read()


refs = list(map(lambda i: i.split("\n"), s.strip().split("\n\n")))[1:-1]

refs_pd = pd.DataFrame(refs, columns=["label", "content", "other"])

refs_pd["author"] = refs_pd["content"].str.extract(r"(.{1})")  # 提取作者
refs_pd["year"] = refs_pd["other"].str.extract(r"(\d{4})").astype(int)  # 提取年份

results = refs_pd.sort_values(by=["author", "year"])

for i, line in enumerate(results.values):
    line_str = "\n".join(line[1:2].tolist()) + line[2]
    line_str = line_str.replace("``", '"').replace(r"{\em ", "").replace("}", "").replace("--", "-")  # 去除多余空格
    print(f"[{i + 1}]", line_str, end="\n")
