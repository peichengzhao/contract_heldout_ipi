"""Example ablation config: set stronger ATTACK_MODEL on OPENAI gateway."""

OPENAI_BASE_URL = "https://your-gateway.example/v1"
OPENAI_API_KEY = "your-openai-compatible-key"

DEEPSEEK_API_KEY = "your-deepseek-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ATTACK_MODEL = "gpt-5.6-sol"
DEFENSE_MODEL = "deepseek-v4-flash"
TASK_AGENT_MODEL = "deepseek-v4-flash"
JUDGE_MODEL = "gpt-5.6-sol"
RULE_JUDGE_MODEL = "gpt-5.6-sol"
OPTIMIZER_MODEL = "deepseek-v4-flash"
