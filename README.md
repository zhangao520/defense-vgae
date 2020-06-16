# DefenseVGAE


## Environment

- CUDA Version: 10.1
- Python 3.6
- GPU: TITAN Xp

## Packages

- torch==1.4.0
- [deeprobust](https://github.com/DSE-MSU/DeepRobust)
- tensorflow-gpu==1.14.0

## Experiments

### On clean data

 ```sh
git clone https://github.com/zhangao520/defense-vgae
cd defense-vgae
python ours_clean.py -d {cora,citeseer,polblogs}
 ```

### Against Nettack

```sh
cd experiments-nettack
./run_defense.sh
```

### Against Mettack

```sh
cd experiments-mettack
python mettack.py --ptb_rate 0.01 --dataset cora
python defense.py --ptb_rate 0.01 --dataset cora -r 10 -j 0.02
```



