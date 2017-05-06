from os import listdir, chdir, getcwd


# Reaches target files in dataset.
def read_files(purpose, _class):
    root_path = getcwd()
    path = 'data/' + purpose + '/' + _class
    chdir(path)

    texts = []
    for file in listdir('.'):
        if file.endswith('.txt'):
            with open(file) as f:
                texts.append(f.read())

    chdir(root_path)
    return texts
