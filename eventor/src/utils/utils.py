import torch
from torch.autograd.variable import Variable


def singleValueToFloatTensor(val, cuda):
    return toTensor([val], cuda, lambda x: torch.FloatTensor(x)) 


def singleValueToLongTensor(val, cuda):
    return toTensor([val], cuda)


def toFloatTensor(val, cuda):
    return toTensor(val, cuda, lambda x: torch.FloatTensor(x))


def toLongTensor(val, cuda):
    return toTensor(val, cuda)


def toTensor(val, cuda, f=lambda x: torch.LongTensor(x)):
    '''
    Embeds the given value into a PyTorch tensor
    :param val: The value
    :param cuda: A flag indicating if the new tensor has to on GPU
    :param f: A conversion function; by default, it converts the value to
    torch.FloatTensor
    '''
    tensor = f(val)
    if cuda:
        tensor = tensor.cuda()
    return tensor


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.Tensor(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor - 1] = 1
    else:
        target[0][floor - 1] = ceil - label
        target[0][ceil - 1] = label - floor
    return target


def null_tensor(size, cudaFlag):
    v = torch.FloatTensor(*size).zero_()
    if cudaFlag:
        v = v.cuda()
    return v


def null_var(size, cudaFlag):
    return Variable(null_tensor(size, cudaFlag))


def print_sizes(vectors: [], *rest):
    if rest is not None:
        print([x.size() if x is not None else None for x in vectors], rest)


def log_masked_softmax(inp, mask, epsilon=0.000001):
    exp_input = inp.exp() + epsilon
    masked_exp_input = exp_input * mask + epsilon
    return (masked_exp_input / masked_exp_input.sum()).log()


def to_var(tensor, cuda):
    var = Variable(tensor)
    if cuda:
        var = var.cuda()
    return var


