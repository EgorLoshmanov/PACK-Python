import numpy as np

x = np.arange(6)
condlist = [x<3, x>3]
choicelist = [-x, x**2]
np.select(condlist, choicelist, 42)