from functools import wraps, partial


class Hook:
    hooks = []

    @staticmethod
    def call_by_name(name, ret=None, *args, **kwargs):
        for hook in Hook.hooks:
            if getattr(hook, 'ENABLE', True) and hasattr(hook, name):
                if ret is None:
                    getattr(hook, name)(*args, **kwargs)
                else:
                    getattr(hook, name)(ret, *args, **kwargs)

    @staticmethod
    def insert(module, priority):
        ins = module()
        ins.__priority__ = priority
        num = len(Hook.hooks)
        for i, v in enumerate(Hook.hooks):
            if ins.__priority__ <= v.__priority__:
                Hook.hooks.insert(i, ins)
                break
        if num == len(Hook.hooks):
            Hook.hooks.append(ins)
        return module

    """ decorator """

    @staticmethod
    def priority(p: int = None):
        return partial(Hook.insert, priority=100 if p is None else p)

    @staticmethod
    def auto(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            Hook.call_by_name(f'{func.__name__}_begin', *args, **kwargs)
            ret = func(*args, **kwargs)
            Hook.call_by_name(f'{func.__name__}_end', ret, *args, **kwargs)
            return ret

        return wrapper


def test():
    @Hook.priority(1)
    class Hook1:
        def main_begin(self):
            print('Hook1 invokes main_begin')

        def main_end(self):
            print('Hook1 invokes main_end')

    @Hook.priority(0)
    class Hook2:
        def main_begin(self):
            print('Hook2 invokes main_begin')

        def main_end(self):
            print('Hook2 invokes main_end')

    @Hook.auto
    def main():
        print('call main')

    print(Hook.hooks)
    main()


if __name__ == '__main__':
    test()
