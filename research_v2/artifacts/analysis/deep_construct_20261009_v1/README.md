# 2026-10-09 深入分析

本目录独立于原投稿包和已发布的主结果。先读 `protocol.md`、保留的初始版本与 `interpretation_amendment.md`，再读最终中文报告。文件中的 `removal` 表示带已知标签的区域占据探针；仅当原生反应追踪核实后，才可把它与实际删除项联系起来。`insertion` 是原始核算残差字段，不能仅凭非负值视为实际插入量。

外部来源固定为 WOMBAT Pattern 1 validation parquet，Hugging Face revision `e86ec35687b0c0ebd1ecc151610a08224d8cf851`。`input_provenance.json` 记录字节数、SHA-256、选择规则、原基准重叠排除和冻结输入。来源声明 CC0 1.0；这轮只复用结构来源，没有复现 WOMBAT 的白盒模型或产生新的活性标签。

256 个分子在生成前选定，失败或无支持者不替换。独立结构不等于独立骨架、活性验证或全化学空间代表性。骨架簇重抽样为事后不确定性敏感性，不能更改原先主判据。

可从当前项目根目录，在 `runtime.json` 所记版本的环境重算。需要既有 MoleculeACE 源表、原 `prepared`/`scored_mutants` 文件及冻结的 CReM 库；这些原文件未改动。

```text
python -B research_v2/scripts/prepare_deep_construct.py
python -B research_v2/scripts/generate_deep_construct_external.py
python -B research_v2/scripts/analyze_deep_construct.py development
python -B research_v2/scripts/analyze_deep_construct.py external
python -B research_v2/scripts/check_deep_construct_native.py
python -B research_v2/scripts/diagnose_deep_construct.py
python -B research_v2/scripts/plot_deep_construct.py
```

首次准备会下载约 158 MB 固定数据，生成会重写本新增目录的外部缓存。应复制本目录或核对最终清单后再重算。分析使用固定 5000 次重抽样和 256 次同区域大小响应置换。响应置换是描述性算子参考，不是已经证明交换性的显著性检验。

全部结果必须同时报告三种概念及不支持结果。主判据和外部支持数不满足时，不可声明独立确认。新增分析、图和论文增补段落尚未自动合并到原 Word/投稿包或公开仓库。
