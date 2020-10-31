

def test():
    for i in range(10):
        yield(i)


test2 = iter(test())


print(next(test2))

print(next(test2))

print(next(test2))
