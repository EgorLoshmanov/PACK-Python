class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16
        
    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False
        
    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return self._count
    
    
    # Ещё один способ изменить атрибут класса
    @count.setter
    def count(self, val):
        self._count = val
        if val <= self._max_count:
            self._counts = val
        else:
            pass
    
    @staticmethod
    def static():
        print('I am function')
    
    @classmethod
    def my_name(cls):
        return cls.__name__
    
    def __add__(self, num):
        """ Сложение с числом """
        return self.count + num
    
    def __mul__(self, num):
        """ Умножение на число """
        return self.count * num
    
    def __lt__(self, num):
        """ Сравнение меньше """
        return self.count < num
    
    def __len__(self):
        """ Получение длины объекта """
        return self.count
    def __lt__(self, other):
        '''Меньше'''
        if isinstance(other, (int,float)):
            return self.count < other
        return self.count < other.count
    
    def __gt__(self, other):
        '''Больше'''
        if isinstance(other, (int,float)):
            return self.count > other
        return self.count > other.count
    
    def __le__(self, other):
        '''Меньше или равно'''
        if isinstance(other, (int,float)):
            return self.count <= other
        return self.count <= other.count
    
    def __ge__(self, other):
        '''Больше или равно'''
        if isinstance(other, (int,float)):
            return self.count >= other
        return self.count >= other.count
    
    def __eq__(self, other):
        '''Равно'''
        if isinstance(other, (int,float)):
            return self.count == other
        return self.count == other.count
    
    def __ne__(self, other):
        '''Не равно'''
        if isinstance(other, (int,float)):
            return self.count != other
        return self.count != other.count
    
    def __iadd__(self, num):
        """ += операция """
        new_count = self.count + num
        if 0 <= new_count <= self._max_count:
            self.count = new_count
        return self
    
    def __isub__(self, num):
        """ -= операция """
        new_count = self.count - num
        if 0 <= new_count <= self._max_count:
            self.count = new_count
        return self
    
    def __imul__(self, num):
        """ *= операция """
        new_count = self.count * num
        if 0 <= new_count <= self._max_count:
            self.count = new_count
        return self