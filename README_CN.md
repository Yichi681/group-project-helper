# README_CN.md（完整版）

```markdown
# Groupwork Ultra Agent（完整版中文说明）

Groupwork Ultra Agent 是一个**本地运行的多智能体 AI 系统**，
用于模拟完整、真实、可控的大学小组作业流程，
使**单个学生**在组织能力与分析深度上接近真实小组。

本项目的重点不是“代写作业”，
而是**自动化小组协作过程**。

---

## 一、项目动机

传统小组作业常见问题：

- 沟通成本高
- 分工不均
- 讨论流于形式
- 大量时间浪费在流程而非思考上

而现代 AI 非常擅长：

- 结构化思考
- 多方案对比
- 反复迭代
- 角色化评估

**本项目的目标**：
> 用多个 AI 角色，模拟一个“高效但理想化”的小组。

---

## 二、核心理念

> **一个人 + 多个 AI = 接近真实小组的能力**

系统中每个 AI：
- 使用不同模型
- 扮演明确角色
- 多轮讨论
- 接受控制器裁决是否继续

---

## 三、智能体角色设计

常见角色包括：

- **Leader（组长）**
  - 汇总内容
  - 去重整合
  - 输出最终版本

- **Researcher（研究者）**
  - 基于课程资料提出可行方案
  - 关注数据来源与现实可操作性

- **Methodologist（方法论）**
  - 检查结构、逻辑、研究设计

- **Critic（批评者）**
  - 指出漏洞与风险
  - 防止“看起来很高级但站不住脚”

- **Editor（编辑）**
  - 提升表达、结构、可读性

- **Red Team（红队，可选）**
  - 从评分和学术风险角度反向攻击方案

- **Controller（裁判模型）**
  - 判断讨论是否已经足够完善
  - 决定是否继续下一轮

---

## 四、多轮讨论机制

每一阶段（选题 / 分工 / 最终稿）都遵循：

1. 各 AI 独立生成方案（结构化 JSON）
2. Leader 合并与去重
3. Controller 判断质量是否达标
4. 若不足 → 带着“聚焦指令”进入下一轮
5. 直到 Controller 判定“可以停”

避免：
- 一轮定生死
- 单模型偏见
- 浅层输出

---

## 五、本地 RAG（资料增强）

支持导入你自己的课程材料：

### 支持格式
- PDF
- PPTX
- DOC / DOCX
- TXT
- Markdown

### 用途
- 课程 slides
- 作业说明
- 评分 rubric
- 背景阅读材料

所有 AI **只能基于已导入资料讨论**，
并被明确禁止编造引用或数据。

---

## 六、模型支持

基于 OpenRouter API，例如：

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `x-ai/grok-4.1-fast`
- `google/gemini-3-pro-preview`

不同角色可以使用不同模型。

---

## 七、整体流程

```text
初始化项目
 → 导入资料
 → 多 AI 选题讨论
 → 多 AI 任务拆解
 → 多 AI 最终稿整合
 → 导出 Markdown / 网页
````

支持 **Autopilot 全自动模式**。

---

## 八、示例命令

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=你的密钥

python main.py init \
  --name "AI Groupwork" \
  --members "Leader,Researcher,Critic,Editor" \
  --leader-model openai/gpt-oss-20b \
  --critic-model openai/gpt-oss-120b \
  --controller-model openai/gpt-oss-120b

python main.py ingest <项目ID>
python main.py propose-topics <项目ID> --mode autopilot
python main.py generate-tasks <项目ID> --mode autopilot
python main.py generate-final <项目ID>
```

---

## 九、学术诚信说明（重要）

本工具用于：

* 结构规划
* 方案对比
* 讨论模拟
* 内容整合

**不等同于代写作业**。

使用者需：

* 自行核查事实
* 合理引用来源
* 遵守课程对 AI 的使用规定

---

## 十、适用场景

* 文科 / 社科 / 商科小组作业
* 课程项目设计
* 报告 / PPT 框架生成
* 学习高质量讨论的结构方式

---

## 十一、免责声明

本项目自动化的是**小组协作过程**，
不是学术责任本身。
最终提交内容由使用者负责。
