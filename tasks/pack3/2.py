class Item:
    def __init__(self, count=3, max_count=16):
        self._max_count = max_count
        self._count = max(0, min(count, max_count))

    def update_count(self, val):
        if 0 <= val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    def __add__(self, num):
        """Сложение с числом"""
        return max(0, min(self._count + num, self._max_count))

    def __sub__(self, num):
        """Вычитание из числа"""
        return max(0, min(self._count - num, self._max_count))
    
    def __mul__(self, num):
        """Умножение на число"""
        return max(0, min(self._count * num, self._max_count))
    
    def __lt__(self, num):
        """ Сравнение меньше """
        return self._count < num
    
    def __le__(self, other):
        """Сравнение меньше или равно"""
        return self._count <= other
    
    def __gt__(self, other):
        """Сравнение больше"""
        return self._count > other

    def __ge__(self, other):
        """Сравнение больше или равно"""
        return self._count >= other
        
    def __eq__(self, other):
        """Проверяем равенство"""
        return self._count == other

    def __iadd__(self, num):
        """+="""
        self._count = max(0, min(self._count + num, self._max_count))
        return self

    def __isub__(self, num):
        """-="""
        self._count = max(0, min(self._count - num, self._max_count))
        return self

    def __imul__(self, num):
        """*="""
        self._count = max(0, min(self._count * num, self._max_count))
        return self
