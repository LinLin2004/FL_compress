import yaml
import itertools
import os
import os.path as osp
import copy

# --- 1. 配置区域 ---

# 你的基础配置文件路径
BASE_CONFIG_PATH = '/home/lhxu/FL_Compress/configs_1026_adamk_revision/cifa_adamk_zerogradient_compress4_p_0.2.yaml'
# 生成的配置文件存放的目录
OUTPUT_DIR = '/home/lhxu/FL_Compress/configs_1113_cifa_adamk'

# 定义你想要调整的超参数网格
# 键 (key): 是一个字符串，用点(.)表示在 YAML 中的嵌套路径
# 值 (value): 是一个列表，包含该超参数所有要尝试的值
# 无攻击一定写 withoutatt，否则生成config不正确
hyperparameter_grid = {
    'byzantine.attack.class_path': [
        f'fl_framework.components.attacks.{atta}' for
            atta in [
                'withoutatt', 
                'random_noise.RandomNoiseAttack',
                'sign_flipping.SignFlipping',
                'zero_gradient.ZeroGradientAttack',
                'label_flipping.LabelFlipping',
                'alie.ALIE',
                'foe.FOE',
                'mimic.Mimic'
            ]
    ],
    "resources" : [

        {
            "gpus" : [1],
        }

    ],
    "aggregator": [
        # {
        #     "class_path": "fl_framework.components.aggregators.compress2.CompressAggregator",
        #     "params": {
        #         "m": 4691,
        #         "r": 2,
        #         "k": 750, 
        #         "byzantine_alpha": 0.25,
        #         "krum_remain": 0.05
        #     },
        # },
        {
            # "class_path": "fl_framework.components.aggregators.krum.KrumAggregator",
            # "params": {"num_byzantine": 4},
            "class_path": "fl_framework.components.aggregators.compress4.CompressAggregator",
            "params": {
                "m": 4691,
                "r": 2,
                "k": 750, 
                "byzantine_alpha": 0.25,
                "krum_remain": 0.2
            },
        },
        {
            "class_path": "fl_framework.components.aggregators.krum.KrumAggregator",
            "params": {"num_byzantine": 4},
        },
        *[
        {"class_path": f"fl_framework.components.aggregators.{agg}"}
        for agg in [
            "geometric_median.GeometricMedianAggregator",
            "mean.MeanAggregator",
            "median.MedianAggregator",
            # "krum.KrumAggregator"
        ]
    ]
    ],
    # 'optimizer.params.tau1': [0.9],
    # 'optimizer.params.tau2': [0.1]
}

base_name = 'cifa_adamk'
save_folder = 'results_cifa_1113_adamk'
def generate_name(combo_dict):
    """
    根据超参数组合生成一个描述性的名称。用于修改config名称，以及实验名和保存路径
    例如: config_lr_0.001_bs_128.yaml
    """
    parts = [base_name]
    for key, value in combo_dict.items():
        # 简化键名，例如 'optimizer.params.lr' -> 'lr'
        if 'tau1' in key or 'tau2' in key:
            # short_key = key.split('.')[-1]
            # parts.append(f"{short_key}_{value}")
            parts.append(f'_{value}')
        if 'aggregator' in key:
            if 'compress' in value['class_path']:
                agg = value['class_path'].split('.')[-2]
            else:
                agg = value['class_path'].split('.')[-1]
            agg_name = agg.replace("Aggregator", "").lower()
            if agg_name == 'geometricmedian':
                agg_name = agg_name.replace('median', '')
            parts.append(f"_{agg_name}")
        elif 'attack' in key:
            att: str = value.split('.')[-1]
            att = att.lower().replace('attack', '')
            parts.append(f'_{att}')
    return "".join(parts)


# --- 脚本主要逻辑 (通常无需修改) ---

def set_nested_value(d, path, value):
    """
    通过点分隔的路径字符串在嵌套字典中设置值。
    例如: set_nested_value(config, 'optimizer.params.lr', 0.01)
    """
    keys = path.split('.')
    current_level = d
    for key in keys[:-1]:
        current_level = current_level.setdefault(key, {})
    current_level[keys[-1]] = value

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载基础配置
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 基础配置文件 '{BASE_CONFIG_PATH}' 未找到。")
        return
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件 '{BASE_CONFIG_PATH}' 时出错: {e}")
        return

    # 准备参数组合
    param_names = list(hyperparameter_grid.keys())
    param_value_lists = list(hyperparameter_grid.values())

    # 使用 itertools.product 生成所有组合
    all_combinations = list(itertools.product(*param_value_lists))
    total_configs = len(all_combinations)
    
    print(f"基础配置: {BASE_CONFIG_PATH}")
    print(f"超参数网格: {len(param_names)} 个参数, 将生成 {total_configs} 个配置文件。")
    print("-" * 30)

    for i, combination in enumerate(all_combinations):
        # 创建基础配置的深拷贝，以免修改原始配置
        new_config = copy.deepcopy(base_config)

        # 将当前组合的值应用到新配置中
        combo_dict = {}
        for param_name, value in zip(param_names, combination):
            if param_name == 'byzantine.attack.class_path' and 'withoutatt' in value:
                set_nested_value(new_config, 'byzantine.num_clients', 0)
            else:
                set_nested_value(new_config, param_name, value)
            combo_dict[param_name] = value
        
        name = generate_name(combo_dict)
        set_nested_value(new_config, 'experiment.name', name)
        set_nested_value(new_config, 'experiment.save_path', osp.join(save_folder, name))

        output_path = os.path.join(OUTPUT_DIR, name+'.yaml')

        # 将新配置写入 YAML 文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # `sort_keys=False` 保持原始顺序
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"({i+1}/{total_configs}) 已生成: {output_path}")
        except IOError as e:
            print(f"错误: 无法写入文件 {output_path}: {e}")
    
    print("-" * 30)
    print(f"成功！总共在 '{OUTPUT_DIR}' 目录中生成了 {total_configs} 个配置文件。")

if __name__ == '__main__':
    main()
