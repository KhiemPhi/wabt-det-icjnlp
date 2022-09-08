# Experiment 0: Best Model Possible: Pairing with Most Similar Comment that Has A Different Label  Must be Topically Aware
#python -u train.py --gpu 1 --batch 40  --learning_rate 1e-4 --epochs 10  -sn "context no augments poly focal loss determinstic pairing"  -c 

#Experiment 1: Effects of Loss Functions  
python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10    -sn "context no augments cb loss"  -c  -l "softmax" 
python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context no augments cross entropy"  -c   -l "cross-entropy"

#Experiment 2: Effects of Augmentation
python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context all augments"  -c -tra -ta 
python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context train augments"  -c  -tra 
python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context test augments"  -c  -ta

#Experiment 3: Effects of Random Pairing + Determinstic Pairing 
#python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context no augments random pairing"  -c  --random

#Experiment 4: Effects of Using Title As Context Padding
#python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context title"  -c --title

#Experiment 5: Topic Agnosticim Whether or Not Topic Agnosticism can be achieved 
#python -u train.py --gpu 2 --batch 80  --learning_rate 1e-4 --epochs 10  -sn "context no augments agnostic"  -c --agnostic

#Experiment 6: Multi-Task Learning with Emotions

#Experiment 7: Multi-Task Learning with Irony

#Experiment 8: Baseline Roberta Base with Irony Pre-Trained
#python -u train.py --gpu 2 --batch 10  --learning_rate 1e-5 --epochs 10  -sn "baseline poly roberta focal loss" 

#Experiment 9: Baseline Roberta Base with Roberta Base Regular
#python -u train.py --gpu 2 --batch 10  --learning_rate 1e-4 --epochs 10  -sn "baseline poly roberta irony focal loss" 



