import argparse

from inferences import attention_scn, pure_attention, pure_scn, tagger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)emantic (C)ompositional (N)ets + Attention] - Generate Caption')

    parser.add_argument('--type', '-t', help='model type')
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument(
        '--model', '-m', default='BEST_checkpoint_pure_attention_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to caption model')
    parser.add_argument('--model_tagger', '-mt',
                        default='BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to tagger model')
    parser.add_argument('--tag_map', '-tm', help='path to tag map JSON')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5,
                        type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    if args.type == 'pure-scn':
        pure_scn.main(args)
    elif args.type == 'attention-scn':
        attention_scn.main(args)
    elif args.type == 'pure-attention':
        pure_attention.main(args)
    else:
        tagger.main(args)
