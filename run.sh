# RACs cell

# nohup python src/pl_trainer.py --cell=RACs --data_name=ptb > lm_ptb_racs.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RACs --data_name=wiki > lm_wiki_racs.log 2>&1 &

# wiki data

# nohup python src/pl_trainer.py --cell=RNN --data_name=wiki --embedding_size=400 --hidden_size=20  --clip=0.25 > lm_wiki_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=MIRNN --data_name=wiki --embedding_size=400 --hidden_size=20  --clip=0.25 > lm_wiki_mirnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=Second --data_name=wiki --embedding_size=400 --hidden_size=20  > lm_wiki_second.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=wiki --rank=20  --clip=0.25 > lm_wiki_tinytnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=wiki --rank=20 > lm_wiki_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RACs --data_name=wiki --embedding_size=400 --hidden_size=20 --clip=0.25 > lm_wiki_racs.log 2>&1 &

# ptb data
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=400 --hidden_size=20 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=MIRNN --data_name=ptb --embedding_size=400 --hidden_size=20 > lm_ptb_mirnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=Second --data_name=ptb --embedding_size=400 --hidden_size=20 --clip=0.25 > lm_ptb_second.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --rank=20 --data_name=ptb --clip=0.25 > lm_ptb_tinytnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --rank=20 --data_name=ptb  > lm_ptb_tinytnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RACs --embedding_size=400 --hidden_size=20 --data_name=ptb --clip=0.25 > lm_ptb_racs.log 2>&1 &

# sweep rank

# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=5 --clip=0.25 > lm_ptb_tnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=10 --clip=0.25 > lm_ptb_tnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=20 --clip=0.25 > lm_ptb_tnlm20.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=25 --clip=0.25 > lm_ptb_tnlm25.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=30 --clip=0.25 > lm_ptb_tnlm30.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=35 --clip=0.25 > lm_ptb_tnlm30.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=40 --clip=0.25 > lm_ptb_tnlm40.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=45 --clip=0.25 > lm_ptb_tnlm45.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=50 --clip=0.25 > lm_ptb_tnlm50.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=ptb --rank=55 --clip=0.25 > lm_ptb_tnlm55.log 2>&1 &

# sweep linear activation

# nohup python src/pl_trainer.py --cell=TinyTNLM --rank=17 --data_name=ptb --clip=0.25 > lm_ptb_tinytnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --rank=17 --data_name=ptb > lm_ptb_tinytnlm2.log 2>&1 &

# nohup python src/pl_trainer.py --cell=TinyTNLM --data_name=wiki --rank=17 --clip=0.25 > lm_wiki_tinytnlm.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=wiki --rank=17 --clip=0.25 > lm_wiki_tinytnlm2.log 2>&1 &

# sweeping for TNLM-Tiny 
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=5 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=10 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=20 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=25 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=30 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=35 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=40 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=45 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=50 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# nohup python src/pl_trainer.py --cell=TinyTNLM2 --data_name=ptb --rank=55 --clip=0.25 > lm_ptb_tinytnlm2.log 2>&1 &
# sweeping rnn 

# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=25 --hidden_size=5 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=10 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=400 --hidden_size=20 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=625 --hidden_size=25 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=900 --hidden_size=30 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=1225 --hidden_size=35 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=1600 --hidden_size=40 --clip=0.25 > lm_ptb_rnn.log 2>&1 &
nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=2025 --hidden_size=45 --clip=0.25 > lm_ptb_rnn_2025_45.log 2>&1 &
nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=2500 --hidden_size=50 --clip=0.25 > lm_ptb_rnn_2500_50.log 2>&1 &

# sweep rnn

# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=5 --clip=0.25 > lm_ptb_rnn_100_5.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=10 --clip=0.25 > lm_ptb_rnn.log_100_10 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=20 --clip=0.25 > lm_ptb_rnn.log_100_20 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=25 --clip=0.25 > lm_ptb_rnn.log_100_25 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=30 --clip=0.25 > lm_ptb_rnn.log_100_30 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=35 --clip=0.25 > lm_ptb_rnn.log_100_35 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=40 --clip=0.25 > lm_ptb_rnn.log_100_40 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=45 --clip=0.25 > lm_ptb_rnn.log_100_45 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=100 --hidden_size=50 --clip=0.25 > lm_ptb_rnn.log_100_50 2>&1 &

# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=5 --clip=0.25 > lm_ptb_rnn_200_5.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=10 --clip=0.25 > lm_ptb_rnn.log_200_10 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=20 --clip=0.25 > lm_ptb_rnn.log_200_20 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=25 --clip=0.25 > lm_ptb_rnn.log_200_25 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=30 --clip=0.25 > lm_ptb_rnn.log_200_30 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=35 --clip=0.25 > lm_ptb_rnn.log_200_35 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=40 --clip=0.25 > lm_ptb_rnn.log_200_40 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=45 --clip=0.25 > lm_ptb_rnn.log_200_45 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=200 --hidden_size=50 --clip=0.25 > lm_ptb_rnn.log_200_50 2>&1 &

# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=5 --clip=0.25 > lm_ptb_rnn_300_5.log 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=10 --clip=0.25 > lm_ptb_rnn.log_300_10 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=20 --clip=0.25 > lm_ptb_rnn.log_300_20 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=25 --clip=0.25 > lm_ptb_rnn.log_300_25 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=30 --clip=0.25 > lm_ptb_rnn.log_300_30 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=35 --clip=0.25 > lm_ptb_rnn.log_300_35 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=40 --clip=0.25 > lm_ptb_rnn.log_300_40 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=45 --clip=0.25 > lm_ptb_rnn.log_300_45 2>&1 &
# nohup python src/pl_trainer.py --cell=RNN --data_name=ptb --embedding_size=300 --hidden_size=50 --clip=0.25 > lm_ptb_rnn.log_300_50 2>&1 &