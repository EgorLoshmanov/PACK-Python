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

class Food(Item):
    """Базовый класс для съедобных объектов"""
    def __init__(self, name, count=1, max_count=10):
        super().__init__(count, max_count)
        self.name = name

    def __str__(self):
        return f"{self.name}(count={self.count})"


# --- Фрукты ---
class Banana(Food):
    def __init__(self, count=1, max_count=10):
        super().__init__("Banana", count, max_count)


class Orange(Food):
    def __init__(self, count=1, max_count=10):
        super().__init__("Orange", count, max_count)


# --- Не фрукты ---
class Beer(Food):
    def __init__(self, count=1, max_count=5):
        super().__init__("Beer", count, max_count)


class Kurochka_s_risom(Food):
    def __init__(self, count=1, max_count=5):
        super().__init__("Kurochka_s_risom", count, max_count)

