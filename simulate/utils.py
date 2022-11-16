__exports__ = ["_option_to_line", "_keyval_to_line"]

def _option_to_line(opt):
    if opt.key and opt.value:
        return f"\t({opt.prefix})\t{opt.key}={opt.value}\n"
    elif opt.key:
        return f"\t({opt.prefix})\t{opt.key}\n"
    else:
        return f"\t({opt.prefix})"
    
def _keyval_to_line(key, value):
    return f"\t{key}={value}\n"
