from pathlib import Path
from easydict import EasyDict
from ..api import wandb

PROJECT_ROOT = Path(__file__).absolute().parents[2]


class Config:
    cfg = None

    @staticmethod
    def merge_custom_cmdline_setting(config, cfg_list):
        """Set config keys via list (e.g., from command line)."""
        from ast import literal_eval
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = k.split('.')
            d = config
            for subkey in key_list[:-1]:
                assert subkey in d, 'NotFoundKey: %s' % subkey
                d = d[subkey]
            subkey = key_list[-1]
            assert subkey in d, 'NotFoundKey: %s' % subkey
            try:
                value = literal_eval(v)
            except:
                value = v

            if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
                key_val_list = value.split(',')
                for src in key_val_list:
                    cur_key, cur_val = src.split(':')
                    val_type = type(d[subkey][cur_key])
                    cur_val = val_type(cur_val)
                    d[subkey][cur_key] = cur_val
            elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
                val_list = value.split(',')
                for k, x in enumerate(val_list):
                    val_list[k] = type(d[subkey][0])(x)
                d[subkey] = val_list
            else:
                assert type(value) == type(d[subkey]), \
                    'type {} does not match original type {}'.format(type(value), type(d[subkey]))
                d[subkey] = value
        return config

    @staticmethod
    def get_output_root(output_root, default=None):
        output_root = Path(output_root) if output_root is not None else default / 'output'
        output_root = output_root if output_root.is_absolute() else default / output_root
        return output_root

    @staticmethod
    def get_experiment_root(tags, output):
        """
        TODO: pass mode := 'eval/eval_tag' to use additional eval_tag
        """
        import datetime
        exp_root = output / tags.dataset / tags.model / tags.experiment / tags.mode
        if 'eval' in tags:
            exp_root /= tags.eval
        log_file = exp_root / f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        ckpt_dir = exp_root / 'ckpt'
        eval_dir = exp_root / 'eval'
        return exp_root, log_file, ckpt_dir, eval_dir

    @staticmethod
    def fromfile_yaml(filename, **kwargs):
        import yaml
        def tag_file_handler(self, node):
            """ !file file1 or !file [file1, file2] """
            ret_dict = {}
            for v in [v for v in node.value] if isinstance(node.value, list) else [node]:
                file_path = Path(self.construct_scalar(v))
                file_path = file_path if file_path.is_absolute() else Path(node.start_mark.name).parent / file_path
                with open(file_path.__str__(), 'r') as f:
                    ret_dict.update(yaml.load(f, Loader=yaml.SafeLoader))
            return ret_dict

        def merge_import_into(config):
            __import_tag__ = 'IMPORT'
            merged_dict = config
            if isinstance(config, (dict, list)):
                for k, v in config.items() if isinstance(config, dict) else enumerate(config):
                    config[k] = merge_import_into(config[k])
                if __import_tag__ in config:
                    merged_dict = config.pop(__import_tag__)
                    merged_dict.update(config)
            return merged_dict

        with open(str(filename), 'r') as f:
            yaml.SafeLoader.add_constructor('!file', tag_file_handler)
            cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_dict = EasyDict(merge_import_into(cfg_dict))
            cfg_dict.CLASS_NAMES = cfg_dict.get('CLASS_NAMES', cfg_dict.DATASET.CLASS_NAMES)
            cfg_dict.DATASET.CLASS_NAMES = cfg_dict.CLASS_NAMES
            cfg_dict.MODEL.CLASS_NAMES = cfg_dict.CLASS_NAMES
        return cfg_dict

    @staticmethod
    def fromfile_py(filepath, config_root=None):
        def import_module(m):
            import sys
            from importlib import import_module
            sys.path.insert(0, config_root)
            sys.dont_write_bytecode = True
            module = import_module(m)
            del sys.modules[m]
            sys.dont_write_bytecode = False
            sys.path.pop(0)
            print(f"load config from module: {m}")
            return module

        import types
        filepath = Path(filepath).absolute()
        config_root = [p for p in filepath.parents if p.name == 'configs'][-1]
        module_path = [p.name for p in filepath.relative_to(config_root).parents.__reversed__()]
        module_path = config_root.name + '.'.join(module_path) + '.' + filepath.stem

        cfg_dict = {
            name: value
            for name, value in import_module(module_path).__dict__.items()
            if not name.startswith('__')
               and not isinstance(value, types.ModuleType)
               and not isinstance(value, types.FunctionType)
               and not isinstance(value, type(Config))
        }
        return cfg_dict

    @staticmethod
    def fromfile(filepath):
        filepath = Path(filepath)
        print(f'load config from file: {filepath}')
        assert filepath.exists() and filepath.is_file()

        config = EasyDict(getattr(Config, f'fromfile_{filepath.suffix[1:]}')(filepath))
        config.config_file = Path(filepath).absolute()
        config.project_root = PROJECT_ROOT
        Config.cfg = config
        return config
