# train a character-level model on enwik8


out_dir = "out-enwik8_baseline_150k"
eval_interval = 5000
eval_iters = 5000
log_interval = 5000  # don't print too too often

# only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'gpt2'
wandb_run_name = 'out-enwik8_baseline_150k'

dataset = "enwik8"
gradient_accumulation_steps = 4
batch_size = 16 
block_size = 512

# GPT-2 model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.3

# optimizer 
learning_rate = 5e-4
max_iters = 150000
warmup_iters = 0  # not super necessary potentially
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99

# progressive loss params
use_prog = False
prog_ratio = 0.2 # progressive training duration ratio
prog_weighting = 'exp' # exp, power, step, or sharp_cutoff

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False
# init_from = 'resume'
# eval_only = True