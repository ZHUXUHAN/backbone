import mxnet as mx


def polt_network(symbol, name):
    mx.viz.plot_network(
        symbol       = symbol,
        title        = name,
        hide_weights = True
    ).view()