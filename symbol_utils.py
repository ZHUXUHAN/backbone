from symbol.fresnet import get_symbol


def get_res18():
    symbol = get_symbol(
        fp16 = 1,
        num_layers  = 18,
        num_classes = 256
    )
    return symbol