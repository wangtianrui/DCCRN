# DCCRN
implementation of "DCCRN-Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"
## how to run
```text
torch==1.7
asteroid==0.3.4
```
* change the "dns_home" of "conf.yml" to the dir of dns-datas
```text
-dns_datas/
    -clean/
    -noise/
    -noisy/
```
* run train.py on pycharm

### test score
```txt
valid set PESQ:
count 2400.000000
mean 3.379963
std 0.821623
min 1.063296
25% 2.748637
50% 3.600925
75% 4.087152
max 4.509835

synthetic test set PESQ:
count 150.000000
mean   2.679080
std    0.708341
min    1.245020
25%    2.104489 
50%    2.679988
75%    3.271556 
max    4.415321   
```

