from io import StringIO
from scipy.io import mmread

m = mmread('/home/fakeheadset/Projects/EulerEasel/Data/bcsstk13.mtx', spmatrix= False);
m2 = m.toarray()
print(m2.shape)