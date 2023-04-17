class replace_attr:
    def __init__(self, obj, attr: str, new_attr) -> None:
        self.obj = obj
        self.attr_name = attr
        self.new_attr = new_attr
        self.old_attr = getattr(self.obj, self.attr_name)

    def __enter__(self) -> None:
        setattr(self.obj, self.attr_name, self.new_attr)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        setattr(self.obj, self.attr_name, self.old_attr)


def merge_dicts(list_of_dict):
    import pandas as pd
    return pd.DataFrame(list(list_of_dict)).to_dict(orient="list")


def test():
    class Model:
        def __init__(self):
            self.name = 'cnn'

    m = Model()
    print(m.name)
    with replace_attr(m, 'name', 'spc'):
        print(m.name)
    print(m.name)


if __name__ == '__main__':
    test()
