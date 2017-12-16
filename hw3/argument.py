def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--max_steps', type=int, default=10000*5000, help='max. number of training steps')
    parser.add_argument('--memory_size', type=int, default=1000000, help='capacity of replay memory')
    parser.add_argument('--learning_start', type=float, default=50000, help='number of steps before training begins')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.00025, help='min. learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='learning rate decay ratio')
    parser.add_argument('--learning_rate_decay_step', type=int, default=50000, help='learning_rate_decay_step')
    parser.add_argument('--rand_epsi', type=float, default=0.05, help='random prob. for epsilon-greedy')
    parser.add_argument('--epsi_final', type=float, default=0.1, help='final learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--target_update_frequency', type=int, default=10000, help='target update frequency')
    parser.add_argument('--testing_frequency', type=int, default=50000, help='testing_frequency')
    parser.add_argument('--train_frequency', type=int, default=4, help='training frequncy')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--double_q', type=int, default=1, help='use double Q learning or not')
    parser.add_argument('--duel_q', type=int, default=0, help='use duel Q learning or not')
    parser.add_argument('--model_path', default=None, help='path to trained model')

    # PG only
    parser.add_argument('--reward_max_len', type=int, default=1000000, help='max. length of reward for normalization')
    parser.add_argument('--reg_param', type=float, default=0.001, help='regularization constant')
    parser.add_argument('--max_episodes', type=int, default=0.001, help='regularization constant')
    
    # Actor-critic
    parser.add_argument('--train_ac', action='store_true', help='whether train actor critic')
    
    
    return parser
