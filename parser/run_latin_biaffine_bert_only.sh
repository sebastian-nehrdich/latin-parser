domain="latin"
word_path="./data/cc.la.300.vec"
declare -i num_epochs=100
declare -i word_dim=300
declare -i set_num_training_samples=100000 # num larger than data set = train on all
start_time=`date +%s`
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
model_path="latin-bert-only"


echo "#################################################################"
echo "Currently BiAFFINE model training in progress..."
echo "#################################################################"
python parser/GraphParser.py --domain $domain --rnn_mode LSTM  --use_bert --bert_outside_gpu --num_epochs $num_epochs --batch_size 32 --hidden_size 512 --arc_space 512 \
--arc_tag_space 128 --num_layers 3 --num_filters 100 \
--set_num_training_samples $set_num_training_samples \
--word_dim $word_dim --char_dim 100 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.001 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 \
--epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding random  --pos_embedding random --word_path $word_path \
--model_path saved_models/$model_path 2>&1 | tee saved_models/base_log.txt

