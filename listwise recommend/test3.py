import pickle
def save_variable(v , filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

filename = 'asd'
x = 1
save_variable(x,filename)
