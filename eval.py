import argparse

from evals import attention_scn, pure_attention, pure_scn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)how (A)ttend (T)ell -- Attention] - Eval Caption')

    parser.add_argument('--type', '-t', help='model type')
    parser.add_argument('--model', '-m', help='pretrained model')
    parser.add_argument('--model_tagger', '-mt',
                        default='BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to tagger model')
    parser.add_argument('--data_folder', '-df',
                        default='./scn_data', help='data folder')
    parser.add_argument(
        '--data_name', '-dn', default='flickr10k_5_cap_per_img_5_min_word_freq', help='data path')
    parser.add_argument('--tag_map', '-tm', help='path to tag map JSON')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-bs', default=5,
                        type=int, help='beam size')
    args = parser.parse_args()

    if args.type == 'pure-scn':
        score = pure_scn.evaluate(args)
    elif args.type == 'attention-scn':
        score = attention_scn.evaluate(args)
    elif args.type == 'pure-attention':
        score = pure_attention.evaluate(args)

    print("\nScore summary @ beam size of % d.\n" %
          (args.beam_size))

    print(score)
