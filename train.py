import argparse

import neptune

from trains import attention_scn, pure_attention, pure_scn, tagger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)how (A)ttend (T)ell - (S)emantic (C)ompositional (N)etworks] - Train Script')

    parser.add_argument('--neptune_key', '-nk', help='neptune key')
    parser.add_argument('--neptune_user', '-nu',
                        default='rayandrew', help='neptune username')
    parser.add_argument('--type', '-t', help='train model type')

    args = parser.parse_args()

    if args.type == 'pure_scn':
        pure_scn.main(args)
    elif args.type == 'attention_scn':
        attention_scn.main(args)
    elif args.type == 'pure_attention':
        pure_attention.main(args)
    else:
        args.type = 'image_tagger'
        tagger.main(args)
