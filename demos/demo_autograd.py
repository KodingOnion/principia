import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.value import Value

print('==============================================')
print('Find local minimum on y=(x-3)^2\n')
x = Value(float(input('Enter a starting guess: ')))
learning_rate = float(input('Enter a learning rate: '))
print('')

for i in range(0,1000):
    y = (x-3)**2
    y.backward()

    print(x.data,y.data)

    x.data += (x.gradient * -learning_rate)
    x.gradient = 0.0