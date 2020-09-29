import os
import nni
import dfhb
import shutil

params = {
          'clip-norm':10,
          'lr': 32,
          'dropout':0.1,
          'relu-dropout':0.1,
          'attention-dropout':0.1,
          'optimizer':"adam",
          'inter_op_parallelism_threads':1,
          'intra_op_parallelism_threads':2,
          'benchmark':0,
          'allow_tf32':0
        }

tuned_params = nni.get_next_parameter() 
params.update(tuned_params) 
t_id = nni.get_trial_id()


data_dir = "./data/processed"
save_dir="./checkpoints/mass/pretraining"
user_dir="./mass"

is_load,load_path,save_path,budget = dfhb.preprocess(t_id,params,save_dir)


seed=1234
max_tokens=2048 # for 16GB GPUs 
update_freq=1
attention_heads=16
embed_dim=1024
ffn_embed_dim=4096
encoder_layers=10
decoder_layers=6
word_mask=0.3

if not os.path.exists(save_path):
    os.makedirs(save_path)


if is_load:
  load_file = os.path.join(load_path,"checkpoint_last.pt")
  os.system("fairseq-train %s --user-dir %s --task xmasked_seq2seq \
    --source-langs en,zh \
    --target-langs en,zh \
      --langs en,zh \
    --arch xtransformer \
      --mass_steps en-en,zh-zh \
      --memt_steps en-zh,zh-en \
      --save-dir %s \
      --output_dir %s\
      --lr-scheduler inverse_sqrt --min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy \
      --lm-bias --lazy-load --seed %d \
      --log-format json \
      --max-tokens %d --update-freq %d \
      --encoder-normalize-before  --decoder-normalize-before \
      --decoder-attention-heads %d --encoder-attention-heads %d \
      --decoder-embed-dim %d --encoder-embed-dim %d \
      --decoder-ffn-embed-dim %d --encoder-ffn-embed-dim %d \
      --encoder-layers %d --decoder-layers %d \
      --max-update 100000000 --max-epoch %d \
      --keep-interval-updates 100 --save-interval-updates 3000  --log-interval 50 \
      --save-interval %d \
      --restore-file %s \
      --share-decoder-input-output-embed \
      --valid-lang-pairs en-zh \
    --word_mask %f \
    --ddp-backend=no_c10d \
    --clip-norm %f --lr %f --dropout %f --attention-dropout %f --relu-dropout %f --optimizer %s \
    --inter %d --intra %d --benchmark %d --allow_tf32 %d"
    %(data_dir,
      user_dir,
      save_path,
      save_path,
      seed,
      max_tokens,update_freq,
      attention_heads,attention_heads,
      embed_dim,embed_dim,
      ffn_embed_dim,ffn_embed_dim,
      encoder_layers,decoder_layers,
      budget,
      budget,
      load_file,
      word_mask,
      params['clip-norm'],params['lr'],params['dropout'],params['attention-dropout'],params['relu-dropout'],params['optimizer'],
      int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32'])))
  shutil.rmtree(load_path)
else:
  os.system("fairseq-train %s --user-dir %s --task xmasked_seq2seq \
    --source-langs en,zh \
    --target-langs en,zh \
      --langs en,zh \
    --arch xtransformer \
      --mass_steps en-en,zh-zh \
      --memt_steps en-zh,zh-en \
      --save-dir %s \
      --output_dir %s\
      --lr-scheduler inverse_sqrt --min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy \
      --lm-bias --lazy-load --seed %d \
      --log-format json \
      --max-tokens %d --update-freq %d \
      --encoder-normalize-before  --decoder-normalize-before \
      --decoder-attention-heads %d --encoder-attention-heads %d \
      --decoder-embed-dim %d --encoder-embed-dim %d \
      --decoder-ffn-embed-dim %d --encoder-ffn-embed-dim %d \
      --encoder-layers %d --decoder-layers %d \
      --max-update 100000000 --max-epoch %d \
      --keep-interval-updates 100 --save-interval-updates 3000  --log-interval 50 \
      --save-interval %d \
      --share-decoder-input-output-embed \
      --valid-lang-pairs en-zh \
    --word_mask %f \
    --ddp-backend=no_c10d \
    --clip-norm %f --lr %f --dropout %f --attention-dropout %f --relu-dropout %f --optimizer %s \
    --inter %d --intra %d --benchmark %d --allow_tf32 %d"
    %(data_dir,
      user_dir,
      save_path,
      save_path,
      seed,
      max_tokens,update_freq,
      attention_heads,attention_heads,
      embed_dim,embed_dim,
      ffn_embed_dim,ffn_embed_dim,
      encoder_layers,decoder_layers,
      budget,
      budget,
      word_mask,
      params['clip-norm'],params['lr'],params['dropout'],params['attention-dropout'],params['relu-dropout'],params['optimizer'],
      int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32'])))

f = open(os.path.join(save_path,"runtime.txt"))
spent_time = float(f.readline())
print(spent_time)


ckpt_path = os.path.join(save_path,"checkpoint_last.pt")
gen_subset = "valid"
result_file = os.path.join(save_path, "result.txt")
target_file = "path_2_target_file"

os.system("fairseq-generate \
--path=%s %s \
--user-dir %s \
-s zh -t en \
--langs en,zh \
--source-langs zh --target-langs en \
--mt_steps zh-en \
--gen-subset %s \
--task xmasked_seq2seq \
--bpe subword_nmt \
--beam 5 --remove-bpe \
> %s"
%(ckpt_path,
  data_dir,
  user_dir,
  gen_subset,
  result_file))


bsf = os.popen("cat %s | grep -P \"^D\" | cut -f 3- | sacrebleu -b %s" % (result_file,target_file))
bs = float(bsf.readline())
print(bs)


report_dict = {'runtime':spent_time,'default':bs,'maximize':['default']}
nni.report_final_result(report_dict)
