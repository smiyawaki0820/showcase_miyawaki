from sklearn.model_selection import train_test_split

a = [str(i) for i in range(100)]

print('hoge')
train, test = train_test_split(a, test_size=0.2)
import ipdb; ipdb.set_trace()
