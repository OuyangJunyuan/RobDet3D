import warnings


class Register(dict):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __setitem__(self, name, module):
        if name in self:
            warnings.warn(f'name {name} has been registered by other module')
        super(Register, self).__setitem__(name, module)

    def register_module(self, *args):
        def register_to(module):
            for name in [module.__name__, *args]: self[name] = module
            return module

        return register_to

    def from_cfg(self, cfg, *args, **kwargs):
        module = self[cfg.get('name', cfg.get('NAME', cfg.get('type')))]
        return module(cfg, *args, **kwargs) if isinstance(module, type) else module

    def from_name(self, name):
        return self[name]


def build_from_cfg(register: Register, cfg, *args, **kwargs):
    return register.from_cfg(cfg, *args, **kwargs)


def test():
    from easydict import EasyDict

    MODELS = Register('model')

    @MODELS.register_module('model1', 'M1')
    class Model1:
        def __init__(self, model_cfg):
            self.name = model_cfg.name
            print(type(self))

    @MODELS.register_module('model2', 'M2')
    class Model2:
        def __init__(self, model_cfg):
            self.name = model_cfg.name
            print(type(self))

    cfg = EasyDict(name='model1')
    m1 = build_from_cfg(MODELS, cfg)
    cfg = EasyDict(name='M2')
    m2 = MODELS.from_cfg(cfg)


if __name__ == '__main__':
    test()
