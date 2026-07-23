import os
import yaml

# 指定 YAML 文件所在目录
yaml_dir = '/home/lhxu/FL_Compress/configs_924_cifa_svrg_zero'  # 请根据实际路径修改

for filename in os.listdir(yaml_dir):
    if filename.endswith('.yaml') or filename.endswith('.yml'):
        filepath = os.path.join(yaml_dir, filename)
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        modified = False

        # 修改 optimizer 学习率
        if 'optimizer' in config and 'params' in config['optimizer']:
            config['optimizer']['params']['lr'] = 0.01
            modified = True

        # 修改 training 轮数
        if 'training' in config:
            config['training']['rounds'] = 100
            modified = True
        
        if config['training'].get('round_steps') != 300:
            config['training']['round_steps'] = 300
            modified = True


        # 保存修改
        if modified:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            print(f"✅ 已更新: {filename}")
        else:
            print(f"⏭️ 无需更新: {filename}")
