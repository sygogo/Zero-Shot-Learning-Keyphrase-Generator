start=1.0
end=20.0
step=0.05
for dataset in 'krapivin' 'nus' 'semeval' 'inspec'; do
  for number in $(seq $start $step $end); do
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    python evaluation_3.py --verbose=False --dataset=$dataset --coefficient=$number >>${dataset}_results_new_2_1.txt
  done
done
