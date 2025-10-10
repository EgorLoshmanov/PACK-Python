class Item:
    """Базовый класс предмета"""
    def __init__(self, count=1, max_count=10):
        self.count = count
        self.max_count = max_count

    def add(self, n):
        if self.count + n <= self.max_count:
            self.count += n
        else:
            raise ValueError("Превышен max_count")

    def remove(self, n):
        if self.count - n >= 0:
            self.count -= n
        else:
            raise ValueError("Количество не может быть меньше 0")

    def __str__(self):
        return f"{self.__class__.__name__}(count={self.count})"


class Food(Item):
    """Съедобные объекты"""
    def __init__(self, name, count=1, max_count=10):
        super().__init__(count, max_count)
        self.name = name

    def __str__(self):
        return f"{self.name}(count={self.count})"


class Inventory:
    """Инвентарь фиксированной длины"""
    def __init__(self, size: int):
        self.size = size
        self.slots = [None] * size  

    def __getitem__(self, idx):
        return self.slots[idx]

    def __setitem__(self, idx, item):
        if not isinstance(item, (Food, type(None))):
            raise TypeError("Можно класть только Food или None")
        self.slots[idx] = item

    def decrease(self, idx, n=1):
        """Уменьшает количество объектов в ячейке"""
        obj = self.slots[idx]
        if obj is None:
            raise ValueError("Ячейка пуста")
        obj.remove(n)
        if obj.count == 0:
            self.slots[idx] = None  # удалить из инвентаря

    def __str__(self):
        return str([str(x) if x else "Empty" for x in self.slots])
