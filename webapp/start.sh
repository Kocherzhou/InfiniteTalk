#!/bin/bash
# 启动数字人 MTV 工作台（家里端，纯 Flask，不加载大模型）。
cd "$(dirname "$0")"

# 载入 .env（去掉行内注释，避免 CRLF 问题）
if [ -f .env ]; then
  set -a
  source <(sed -E 's/[[:space:]]+#.*$//; s/\r$//' .env)
  set +a
fi

PORT="${PORT:-28600}"
echo "🎤 启动 数字人 MTV 工作台..."
echo "   AUTH_TOKEN:   $([ -n "$AUTH_TOKEN" ] && echo '✓ 已设置' || echo '✗ 未设置(无登录)')"
echo "   WORKER_TOKEN: $([ -n "$WORKER_TOKEN" ] && echo '✓ 已设置' || echo '✗ 未设置')"
echo "   URL: http://localhost:${PORT}"

# 用系统 python3（deps: flask, python-dotenv；与 video-subtitle 同环境）
exec /usr/bin/python3 app.py
