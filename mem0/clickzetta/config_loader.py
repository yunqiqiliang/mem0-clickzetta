#!/usr/bin/env python3
"""
通用的环境配置加载工具

支持多种配置文件位置：
1. 相对路径：./server/.env
2. 环境变量：CLICKZETTA_ENV_FILE
3. 默认位置：当前目录下的 .env
"""

import os
from typing import Dict, Optional

def load_env_config(env_file_path: Optional[str] = None) -> Dict[str, str]:
    """
    加载环境配置文件

    Args:
        env_file_path: 可选的配置文件路径

    Returns:
        配置字典
    """
    config = {}

    # 确定配置文件路径的优先级
    possible_paths = []

    # 1. 如果指定了路径，优先使用
    if env_file_path:
        possible_paths.append(env_file_path)

    # 2. 检查环境变量
    env_var_path = os.getenv('CLICKZETTA_ENV_FILE')
    if env_var_path:
        possible_paths.append(env_var_path)

    # 3. 相对路径（从脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # 向上两级到项目根目录
    possible_paths.extend([
        os.path.join(project_root, 'server', '.env'),
        os.path.join(script_dir, 'server', '.env'),
        os.path.join(script_dir, '.env'),
        os.path.join(os.getcwd(), 'server', '.env'),
        os.path.join(os.getcwd(), '.env')
    ])

    # 尝试加载第一个存在的文件
    for env_file in possible_paths:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            try:
                                key, value = line.split('=', 1)
                                config[key.strip()] = value.strip()
                            except ValueError:
                                print(f"Warning: Invalid line {line_num} in {env_file}: {line}")

                print(f"✅ Loaded configuration from: {env_file}")
                return config

            except Exception as e:
                print(f"❌ Error reading {env_file}: {e}")
                continue

    print("❌ No valid configuration file found!")
    print("Searched paths:")
    for path in possible_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {path}")

    return config

def validate_clickzetta_config(config: Dict[str, str]) -> bool:
    """
    验证 ClickZetta 配置是否完整

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_keys = [
        'CLICKZETTA_SERVICE',
        'CLICKZETTA_INSTANCE',
        'CLICKZETTA_WORKSPACE',
        'CLICKZETTA_SCHEMA',
        'CLICKZETTA_USERNAME',
        'CLICKZETTA_PASSWORD',
        'CLICKZETTA_VCLUSTER'
    ]

    missing_keys = []
    for key in required_keys:
        if not config.get(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"❌ Missing required configuration keys: {', '.join(missing_keys)}")
        return False

    print("✅ ClickZetta configuration is valid")
    return True

def get_clickzetta_config(env_file_path: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    获取并验证 ClickZetta 配置

    Args:
        env_file_path: 可选的配置文件路径

    Returns:
        有效的配置字典，如果无效则返回 None
    """
    config = load_env_config(env_file_path)

    if not config:
        return None

    if not validate_clickzetta_config(config):
        return None

    return config

if __name__ == "__main__":
    print("=" * 60)
    print("ClickZetta 配置加载测试")
    print("=" * 60)

    # 测试配置加载
    config = get_clickzetta_config()

    if config:
        print("\n📋 加载的配置:")
        for key, value in config.items():
            if 'PASSWORD' in key or 'KEY' in key:
                # 隐藏敏感信息
                masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '***'
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: {value}")

        print(f"\n🎉 配置加载成功！找到 {len(config)} 个配置项")
    else:
        print("\n❌ 配置加载失败")

        print("\n💡 解决方案:")
        print("1. 确保 server/.env 文件存在")
        print("2. 或者设置环境变量 CLICKZETTA_ENV_FILE 指向配置文件")
        print("3. 或者在当前目录创建 .env 文件")

        print("\n📝 配置文件示例内容:")
        print("""
# ClickZetta Configuration
CLICKZETTA_SERVICE=your-service.clickzetta.com
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_VCLUSTER=your-vcluster

# Optional
CLICKZETTA_CONNECTION_TIMEOUT=30
CLICKZETTA_QUERY_TIMEOUT=300
DASHSCOPE_API_KEY=your-dashscope-key
        """.strip())