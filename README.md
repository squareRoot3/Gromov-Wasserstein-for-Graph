# A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in Graph Data

This is a Python implementation of 

>  A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in Graph Data
>
>  *Jiajin Li, Jianheng Tang, Lemin Kong, Huikang Liu, Jia Li, Anthony Man-Cho So, Jose Blanchet*
>
>  [**ICLR 2023**](https://openreview.net/pdf?id=0jxPyVWmiiF)


Dependencies
----------------------
```
pip install -r requirements.txt
```

Graph Alignment Experiments (Section 4.2)
--------------------------------
```
bash reproduce_BAPG.sh
bash reproduce_eBPG.sh
bash reproduce_others.sh
```

Graph Partition Experiments (Section 4.3)
--------------------------------
```
python graph_partition_amazon.py
python graph_partition_eu.py
python graph_partition_village.py
python graph_partition_wiki.py
```




