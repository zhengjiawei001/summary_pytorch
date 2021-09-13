import argparse


def common_args(parser):
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--unk', type=int, default=1)
    parser.add_argument('--bos', type=int, default=2)
    parser.add_argument('--eos', type=int, default=3)
    parser.add_argument('--max_article_length', type=int, default=149)
    parser.add_argument('--max_summary_length', type=int, default=30)
    parser.add_argument('--train_idxs', type=str, default='/data/Xuhuipeng/LCSTS/train_idxs.pt')
    parser.add_argument('--valid_idxs', type=str, default='/data/Xuhuipeng/LCSTS/valid_idxs.pt')
    parser.add_argument('--pretrain_train_idxs', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_train_idxs.pt')
    parser.add_argument('--pretrain_valid_idxs', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_valid_idxs.pt')
    parser.add_argument('--pretrain_test_idxs', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_test_idxs.pt')
    parser.add_argument('--debug_train_idxs', type=str, default='/data/Xuhuipeng/LCSTS/debug_train_idxs.pt')
    parser.add_argument('--debug_valid_idxs', type=str, default='/data/Xuhuipeng/LCSTS/debug_valid_idxs.pt')
    parser.add_argument('--test_idxs', type=str, default='/data/Xuhuipeng/LCSTS/test_idxs.pt')
    parser.add_argument('--debug_token2id', type=str, default='/data/Xuhuipeng/LCSTS/debug_token2id.json')
    parser.add_argument('--debug_id2token', type=str, default='/data/Xuhuipeng/LCSTS/debug_id2token.json')
    parser.add_argument('--pretrain_token2id', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_token2id.json')
    parser.add_argument('--pretrain_id2token', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_id2token.json')
    parser.add_argument('--pretrain_embedding_file', type=str, default='/data/Xuhuipeng/LCSTS/pretrain_embedding.pt')
    parser.add_argument('--pretrain_file', type=str, default='/data/Xuhuipeng/LCSTS/sgns.weibo.bigram-char.bz2')
    parser.add_argument('--vocab_size', type=int, default=4000)


def setup_args():
    parser = argparse.ArgumentParser(description='for set up')

    common_args(parser)

    parser.add_argument('--train_file', type=str, default='/data/Xuhuipeng/LCSTS/PART_I.txt')
    parser.add_argument('--extract_folder', type=str, default='/data/Xuhuipeng/LCSTS/')
    parser.add_argument('--train_zip_file', type=str, default='/data/Xuhuipeng/LCSTS/PART_I.zip')
    parser.add_argument('--test_file', type=str, default='/data/Xuhuipeng/LCSTS/PART_III.txt')
    parser.add_argument('--valid_split', type=float, default=0.01)

    args = parser.parse_args()

    return args


def train_args():
    parser = argparse.ArgumentParser(description='training args')

    common_args(parser)
    train_test_args(parser)

    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--is_coverage', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--init_accumulator_val', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lambda_', type=float, default=1.0, help='coverage loss weight')
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--eval_steps', type=int, default=480000)
    parser.add_argument('--max_checkpoints', type=int, default=5)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--metric_name', type=str, default='loss',
                        choices=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'loss'])

    args = parser.parse_args()

    return args


def train_test_args(parser):
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='/output/')
    parser.add_argument('--num_workers', type=int, default=4)


def test_args():
    parser = argparse.ArgumentParser(description='test args')

    common_args(parser)
    train_test_args(parser)
    parser.add_argument('--best_model', type=str, default='/data/Xuhuipeng/LCSTS/best.pth.tar')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--visiual_prob', type=int, default=0.2)

    args = parser.parse_args()

    return args
