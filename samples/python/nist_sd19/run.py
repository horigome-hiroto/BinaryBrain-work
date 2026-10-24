"""Short entry point retaining the original per-network experiment drivers."""
import argparse
import importlib
import sys

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model',choices=('fpga_magazine','g_research'),required=True)
    args,rest=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+rest
    module='fpga_magazin_network_samepadding' if args.model=='fpga_magazine' else 'g_research_miou_replicate_fixed'
    importlib.import_module(module).main()

if __name__=='__main__':main()
