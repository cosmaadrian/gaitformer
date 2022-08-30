

class DatasetCache(object):
    def __init__(self, manager):
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        is_it = str(key) in self._dict
        return is_it

    def reset(self):
        self._dict.clear()

    def get(self, key):
        return self._dict[str(key)]

    def put(self, key, value):
        if str(key) in self._dict:
            return False

        self._dict[str(key)] = value

        return True