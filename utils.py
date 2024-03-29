import torch.nn.init as init


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)


def write_to_file(log, filename):
    f = open(filename, "w")
    f.write(log)
    f.close()
