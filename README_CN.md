# AI Group Team Simulator（AI 虚拟小组作业系统）

这是一个**多模型 AI 协作工具**，用于模拟一个完整的“小组作业小组”，让**单人用户**通过多个不同大模型的协作，获得接近真实小组讨论的能力。

本项目的目标不是“代写作业”，而是帮助你：
- 理清作业结构
- 发现逻辑漏洞与缺失部分
- 设计可执行的任务方案
- 通过多轮讨论生成一份高质量 Markdown 作业大纲

---

## ✨ 核心思想

> **一个人 + 多个 AI = 虚拟小组**

系统中，不同 AI 扮演不同“组员角色”：

- **Leader（组长）**：负责整体结构、逻辑整合  
- **Critic（挑刺者）**：负责找问题、质疑和改进建议  
- **Researcher（研究员）**：负责方法、数据来源、执行细节  

流程是：
1. 所有模型读取相同的课程背景与作业要求  
2. 各自独立给出方案  
3. 多轮互相点评、重写  
4. 由“组长模型”整合成最终 Markdown 版本  

---

## 📁 项目结构说明

````

.
├── ai_group_team.py
├── Background_Information/
│   ├── syllabus.md
│   ├── lecture_notes.txt
│   └── ...
├── Assignment_Requirement/
│   ├── assignment_prompt.txt
│   ├── rubric.md
│   └── ...
├── README.md
└── README_CN.md

````

### Background_Information（课程背景）
放课程层面的信息，例如：
- 课程大纲 / syllabus  
- 老师 PPT、lecture notes  
- 核心概念、方法论说明  

### Assignment_Requirement（作业要求）
放具体作业说明，例如：
- 作业题目与要求  
- 评分标准（rubric）  
- 指定结构模板  
- 字数、格式、截止时间  

脚本会自动读取这两个文件夹中所有可读文本文件（`.txt` / `.md`）。

---

## 🔧 环境要求

- Python 3.9+
- 仅需一个依赖：
```bash
pip install requests
````

---

## 🔑 API Key 配置

支持**所有 OpenAI 兼容接口**。

API Key 优先级：

1. 命令行 `--api-key`
2. 环境变量 `LLM_API_KEY`
3. `OPENAI_API_KEY`
4. `OPENROUTER_API_KEY`

推荐设置方式：

```bash
export LLM_API_KEY="你的API Key"
```

---

## 🚀 快速开始（OpenRouter 示例）

默认使用三种模型模拟小组：

* `openai/gpt-5.2`（组长）
* `x-ai/grok-4.1-fast`（挑刺）
* `google/gemini-3-pro-preview`（研究）

```bash
python ai_group_team.py \
  --api-base https://openrouter.ai/api/v1 \
  --rounds 3 \
  --out Final_Group_Report.md
```

运行完成后会生成：

* `Final_Group_Report.md`（最终综合 Markdown）

---

## ⚙️ 自定义模型 / 接口

### 使用 OpenAI 官方接口

```bash
python ai_group_team.py \
  --api-base https://api.openai.com/v1 \
  --leader-model gpt-4.1 \
  --critic-model gpt-4.1-mini \
  --researcher-model gpt-4.1-mini \
  --rounds 2 \
  --out Final_Group_Report.md
```

### 使用自建或代理接口

```bash
python ai_group_team.py \
  --api-base https://your-gateway.example.com/v1 \
  --api-key YOUR_KEY \
  --leader-model model-a \
  --critic-model model-b \
  --researcher-model model-c \
  --rounds 4 \
  --out result.md
```

---

## 🧠 推荐使用流程

1. 整理课程资料 → 放入 `Background_Information/`
2. 整理作业说明 → 放入 `Assignment_Requirement/`
3. 运行脚本生成 Markdown 作业方案
4. **人工处理**：

   * 核查事实
   * 补充文献引用
   * 用自己的语言重写
   * 转成正式报告或 PPT

---

## ⚠️ 学术诚信说明（非常重要）

本工具：

* ❌ 不能替代你的思考
* ❌ 不保证内容完全正确
* ❌ 不应直接作为原创作业提交

它适合用于：

* 头脑风暴
* 作业结构规划
* 多角度自我审查

请务必遵守你所在学校和课程对 AI 使用的相关规定。

---

## 📜 许可说明

本项目仅用于学习与研究辅助，不提供任何形式的担保。
