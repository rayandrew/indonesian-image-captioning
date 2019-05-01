import argparse

from evals import attention_scn, pure_attention, pure_scn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)how (A)ttend (T)ell -- Attention] - Generate Caption')

    parser.add_argument('--model', '-m', help='train model')
    parser.add_argument('--beam_size', '-bs', default=5,
                        type=int, help='beam size')
    args = parser.parse_args()

    if args.model == 'pure-scn':
        score = pure_scn.evaluate(args.beam_size)
    elif args.model == 'attention-scn':
        score = attention_scn.evaluate(args.beam_size)
    elif args.model == 'pure-attention':
        score = pure_attention.evaluate(args.beam_size)

    print("\nScore summary @ beam size of % d.\n" %
          (args.beam_size))

    print(score)
