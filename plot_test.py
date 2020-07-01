from symbol_utils import *
from plot import polt_network

symbol_dict = {
    'res18': get_res18()
}

if __name__ == '__main__':
    polt_network(symbol_dict['res18'], 'res18')
