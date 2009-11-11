
python2.6 -OO cerberus.py --log kddfull_train.log --gamma 0.6 --C 10 --EPS 0.01 --model_out kddfull.mdl datasets/kddcup.names datasets/kddcup_10-normalised.txt.gz > kdd_train.results 
python2.6 -OO cerberus.py --log kddfull_test.log --model_in kddfull.mdl datasets/kddcup.names datasets/corrected_out-normalised.txt.gz > kdd_test.results


python2.6 -OO cerberus.py --log kddfull_train.log --gamma 0.6 --C 10 --EPS 0.01 --model_out kddfull.mdl datasets/kddcup.names datasets/kddcup_10-normalised.txt.gz >> kdd_train.results 
python2.6 -OO cerberus.py --log kddfull_test.log --model_in kddfull.mdl datasets/kddcup.names datasets/corrected_out-normalised.txt.gz >> kdd_test.results


python2.6 -OO cerberus.py --log kddfull_train.log --gamma 0.6 --C 10 --EPS 0.01 --model_out kddfull.mdl datasets/kddcup.names datasets/kddcup_10-normalised.txt.gz >> kdd_train.results 
python2.6 -OO cerberus.py --log kddfull_test.log --model_in kddfull.mdl datasets/kddcup.names datasets/corrected_out-normalised.txt.gz >> kdd_test.results
