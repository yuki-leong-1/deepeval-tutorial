# Pipeline Metrics — 哪些能 measure?能不能一起?

针对图里 "Metric Runner" 的每一个指标:**能不能测量,用什么方法,能不能放一起 `evaluate()`。**

## 一句话结论

- **能不能一起 measure 的判断规则**:同一次 `evaluate(test_cases=[tc], metrics=[...])` 里,所有 metric 都跑在**同一个 `LLMTestCase`** 上。只要某个 metric 是单轮 `BaseMetric`,且它读的 `actual_output / expected_output` 含义一致,就能放一起。
- 所以**按流水线阶段分文件**:翻译 / STT / TTS / 系统,每个阶段内部能一起测;**跨阶段不能**(因为同一个 `actual_output` 在翻译阶段是译文、在 STT 阶段是转写文本,含义不同,放一起会算出垃圾分数)。
- DeepEval 没有内置 WER/chrF++/COMET,但**任何打分器都能包成自定义 `BaseMetric`**,包完就能和 GEval 一起跑。
- **系统指标**(延迟/成本/失败率)不是"给输出内容打分",大多要靠埋点;按 + 单轮门槛 / 聚合 拆开。

## 逐项对照表

| 图中指标 | 能 measure? | 方法 / 库 | 放哪个文件(一起) | 要 API key? |
|---|---|---|---|---|
| **COMET / XCOMET** | ✅ 可,较重 | `unbabel-comet`(神经网络,需下载 ~2GB 模型)→ 包成 `CometMetric` | `01_translation_metrics.py`(默认关,`USE_COMET=1` 开) | 否(但要装重依赖) |
| **chrF++** | ✅ 可 | `sacrebleu.sentence_chrf(word_order=2)` → `ChrfppMetric` | `01`(和下面一起) | 否 |
| **LLM judge via DeepEval** | ✅ 可 | DeepEval 原生 `GEval` | `01` | **是** |
| **number accuracy** | ✅ 可 | 正则抽数字比对 → `NumberAccuracyMetric`(确定性) | `01` | 否 |
| **unit accuracy** | ✅ 可 | 单位词典比对 → `UnitAccuracyMetric`(确定性) | `01` | 否 |
| **drug accuracy** | ✅ 可 | 语义判断 → `GEval`(临床判官) | `01` | **是** |
| **negation accuracy** | ✅ 可 | 语义判断 → `GEval` | `01` | **是** |
| **frequency accuracy** | ✅ 可 | 语义判断 → `GEval` | `01` | **是** |
| **critical fact accuracy** | ✅ 可 | 语义判断 → `GEval` | `01` | **是** |
| **WER** | ✅ 可 | `jiwer.wer` → `WERMetric` | `02_stt_metrics.py` | 否 |
| **CER** | ✅ 可 | `jiwer.cer` → `CERMetric` | `02` | 否 |
| **medical term WER** | ✅ 可 | 过滤医学词后算 WER → `MedicalTermWERMetric` | `02` | 否 |
| **number WER** | ✅ 可 | 只保留数字后算 WER → `NumberWERMetric` | `02` | 否 |
| **ASR loopback WER** | ✅ 可 | TTS→ASR 回环后 `jiwer.wer` → `ASRLoopbackWERMetric` | `03_tts_audio_metrics.py` | 否 |
| **pronunciation flags** | ⚠️ 部分 | 检测要靠**音频模型/强制对齐**(DeepEval 外);分数由你算好后,用 `metadata` 传进来当门槛 | `03`(读 `metadata`) | 否 |
| **intelligibility** | ⚠️ 部分 | 同上,要 **MOS/STOI 预测模型**;算好的 0–1 分用 `metadata` 传进来 | `03`(读 `metadata`) | 否 |
| **time to first audio** | ✅ 可(门槛) | 埋点计时 → `metadata['ttfa_s']` → `TimeToFirstAudioMetric` | `04_system_metrics.py`(A 部分) | 否 |
| **end-to-end turn latency** | ✅ 可(门槛) | 埋点计时 → `LLMTestCase.completion_time` → `MaxLatencyMetric` | `04`(A 部分) | 否 |
| **cost per minute** | ❌ 非单轮 metric | **聚合量** = 总成本 / 总音频分钟,用普通 Python 算 | `04`(B 部分,**单独**) | 否 |
| **failure rate** | ❌ 非单轮 metric | **聚合量** = 失败轮数 / 总轮数,用普通 Python 算 | `04`(B 部分,**单独**) | 否 |

## 为什么这样分组(放一起 vs 分开)

1. **翻译指标 + 医疗安全指标** → 都在评判**同一条译文**(`input`=原文,`actual_output`=机器译文,`expected_output`=参考译文)。所以共用一个 `LLMTestCase`,在 `01` 里一次 `evaluate()` 全部测完。
2. **STT 指标** → 评判**转写文本**(`expected_output`=人工转写,`actual_output`=ASR 结果)。和翻译的 `actual_output` 含义不同 → 必须单独放 `02`。`02` 全部确定性、**不需要 key**。
3. **TTS/音频指标** → 评判**合成语音**。回环 WER 能从文本算;发音/可懂度的"打分"在音频模型里,DeepEval 只能接收你算好的数值(经 `metadata`)→ 一起报告但只有回环 WER 是本文件算的,放 `03`。
4. **系统指标** → 不是给"输出内容"打分:
   - **每轮**有值的(延迟、首音时间)→ 可包成 DeepEval 门槛 metric,一起测(`04` A 部分)。
   - **跨多轮**才有意义的(每分钟成本、失败率)→ 没有"单条 test case 的分数",所以**不是** DeepEval metric,用普通 Python 聚合(`04` B 部分,故意分开)。

> 关键限制:`evaluate()` 会让**每个 metric 跑在每条 test case 上**。如果把翻译和 STT 的 test case 混在一次调用里,`WERMetric` 也会去算翻译那条 → 得到无意义的分数。这就是必须按阶段分文件的根本原因。

## 方向(direction)别搞反

| 类型 | 例子 | 判定 |
|---|---|---|
| 越高越好(accuracy/质量) | chrF++、COMET、number/unit accuracy、GEval、intelligibility | `score >= threshold` 才 pass |
| 越低越好(错误率) | WER、CER、medical-term/number WER、loopback WER、pronunciation flag rate、latency | `score <= threshold` 才 pass |

## 怎么跑

```bash
# 先装好(已在 pyproject):deepeval, jiwer, sacrebleu, python-dotenv
uv sync

# 无需 key,直接跑:
uv run python pipeline_metrics/02_stt_metrics.py
uv run python pipeline_metrics/03_tts_audio_metrics.py
uv run python pipeline_metrics/04_system_metrics.py

# 需要 OPENAI_API_KEY(GEval 临床判官):先把 key 写进 .env
uv run python pipeline_metrics/01_translation_metrics.py
# 想顺带跑 COMET(需 `uv add unbabel-comet`,会下大模型):
USE_COMET=1 uv run python pipeline_metrics/01_translation_metrics.py
```

> 词典是占位用的小样本:`_common.py` 里的 `UNIT_LEXICON`、`MEDICAL_TERM_LEXICON` 请换成你自己的临床词表;医学语义类(drug/negation/frequency/critical-fact)用 GEval 比写死规则更鲁棒、且跨语言通用。
