def condense_range(tensor):
    return (tensor + 1) * 0.5

def expand_range(tensor):
    return (tensor * 2) - 1