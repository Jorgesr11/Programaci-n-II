import pandas as pd
import numpy as np

s = pd.Series(['Matemáticas', 'Historia', 'Economía', 'Programación', 'Inglés'], dtype='string')
print(s)
s = pd.Series({'Matemáticas': 6.0, 'Economía': 4.5, 'Programación': 8.5})
print (s)
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(s.size)
print(s.index)
s = pd.Series([1, 2, 3, 4])
print(s.apply(np.log))
s = pd.Series(['a', 'b', 'c'])  
print(s.apply(str.upper))  
s = pd.Series({'Matemáticas': 6.0, 'Economía': 4.5, 'Programación': 8.5})
print(s[s > 5])
s = pd.Series({'Matemáticas': 6.0, 'Economía': 4.5, 'Programación': 8.5})
print(s.sort_values())
s=pd.Series(['a', 'b', None, 'c', np.NaN, 'd’])
print(s)
