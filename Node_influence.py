from datetime import datetime
from functools import reduce
import math
from typing import List, Dict
from collections import Counter


class _2DSE:

    def __init__(self, edges):
        self.node_idx:Dict[int, int] = {}
        self.nodes = []
        self.comm_vol: List[float] = []
        self.comm_cut: List[float] = []
        self.node_degree: List[float] = []
        self.node_deg_in:List[float] = []
        self.node_comm:List[int] = []
        self.sorted_nodes: List[int, float] = []
        self.stop_move:List[float] = []
        self.node_inf:List[float] = []
        self.vol = 0.
        self.node_num = 0
        self.node_deg_in_overlap = {}
        self.adj: List[Dict[int, float]] = []
        # build graph from edges
        for a, b in edges:
            if a not in self.node_idx:
                self.node_num += 1
                self.node_idx[a] = len(self.nodes)
                self.nodes.append(a)
                self.adj.append({})

                self.node_degree.append(0.)
                self.node_comm.append(self.node_idx[a])

            if b not in self.node_idx:
                self.node_num += 1
                self.node_idx[b] = len(self.nodes)
                self.nodes.append(b)
                self.adj.append({})

                self.node_degree.append(0.)
                self.node_comm.append(self.node_idx[b])

            i = self.node_idx[a]
            j = self.node_idx[b]
            if i != j:
                self.adj[i][j] = 1.
                self.adj[j][i] = 1.

                self.node_degree[i] += 1.
                self.node_degree[j] += 1.

                self.vol += 2.

        self.comm_vol = self.node_degree.copy()
        self.comm_cut = self.node_degree.copy()
        # self.sorted_nodes = sorted(enumerate(self.node_degree), key=lambda it: it[1], reverse=True)
        self.sorted_nodes = [(x,node) for x,node in enumerate(self.node_degree)]
        self.comm_num = self.node_num
        self.node_deg_in = [0.] * self.node_num
        self.stop_move = self.node_deg_in.copy()

        self.node_inf = self.node_deg_in.copy()
        del self.node_idx

    def SE_2D(self, d, g, v, n_in:float =0, leave=True,cal_node_inf=False):

        '''
        keep state
        '''
        if n_in < 1e-8 and math.fabs(d-g) < 1e-8 and math.fabs(d-v) <1e-8:
            return  0.

        if leave:
            '''
            leave community
            '''

            _g = g - d + (2 * n_in)
            _v = v - d

        else:

            '''
            join community
            '''
            _g = g
            _v = v

            g = _g + d - (2 * n_in)
            v = _v + d

        if  cal_node_inf:
            se = - (_g / self.vol) * math.log2(_v / self.vol) + (g / self.vol) * math.log2(v / self.vol) + (
                    _v / self.vol) * math.log2(_v / v) + (d / self.vol) * math.log2(d / v)
        else:
            se =  - (_g/self.vol)*math.log2(_v/self.vol) + (g/self.vol)*math.log2(v/self.vol) + (_v/self.vol)*math.log2(_v/v) +(d/self.vol)*math.log2(self.vol/v)

        return se

    def be_new_comm(self, node,d):
        # update clust C_k
        comm = self.node_comm[node]
        self.comm_vol[comm] -= d
        self.comm_cut[comm] = self.comm_cut[comm] + 2 * self.node_deg_in[node] - d

        self.comm_num +=1
        # new cluster {x}
        self.node_comm[node] = self.comm_num
        self.comm_vol[self.comm_num] = d
        self.comm_cut[self.comm_num] = d
        self.node_deg_in[node] = 0

        for r_node, w in self.adj[node].items():
            if self.node_comm[r_node] == comm:
                self.node_deg_in[r_node] -= w

    def to_new_comm(self, node, t_comm, node_in,d):
        s_comm = self.node_comm[node]
        self.comm_vol[s_comm] -= d
        self.comm_cut[s_comm] = self.comm_cut[s_comm] + 2 * self.node_deg_in[node] - d

        self.node_comm[node] = t_comm
        self.comm_vol[t_comm] += d
        self.comm_cut[t_comm] = self.comm_cut[t_comm] - 2 * node_in + d
        self.node_deg_in[node] = node_in

        for r_node, w in self.adj[node].items():
            if self.node_comm[r_node] == s_comm:
                self.node_deg_in[r_node] -= w
            if self.node_comm[r_node] == t_comm:
                self.node_deg_in[r_node] += w

    def fit(self, max_item=10,patience=1,verbose=False):
        if verbose:
            start = datetime.now()
            print("-" * 30)
            print('Start game')
        for it in range(max_item):
            if verbose:
                print("-" * 20)
                ind = 0
            delta_sum = 0
            move = 0
            for node, d in self.sorted_nodes:
                if verbose:
                    ind += 1
                    print('\rgaming {}: {:.2f}%'.format(it, ind/self.node_num*100), end="", flush=True)

                if self.stop_move[node] > patience:
                    continue


                s_comm = self.node_comm[node]

                se_be_new_comm = self.SE_2D(d, self.comm_cut[s_comm], self.comm_vol[s_comm], self.node_deg_in[node])

                adj_div = {}
                for r_node, w in self.adj[node].items():
                    r_comm = self.node_comm[r_node]
                    if r_comm != s_comm:
                        if r_comm not in adj_div:
                            adj_div[r_comm] = 0.
                        adj_div[r_comm] += w


                node_in = None
                delta_min = 0
                t_comm = None
                for new_comm, n_in in adj_div.items():
                    tar_g = self.comm_cut[new_comm]
                    tar_v = self.comm_vol[new_comm]
                    se_to_new_comm = se_be_new_comm - self.SE_2D(d, tar_g, tar_v, n_in, False)

                    if se_to_new_comm < delta_min:
                        delta_min = se_to_new_comm
                        t_comm = new_comm
                        node_in = n_in



                if delta_min < 0:
                    delta_sum += delta_min
                    self.to_new_comm(node, t_comm, node_in, d)
                    move +=1

                elif se_be_new_comm < 0:
                    delta_sum += se_be_new_comm
                    self.be_new_comm(node, d)
                    move +=1

                else:
                    self.stop_move[node] += 1

            if verbose:
                end = datetime.now()
                print("\n")
                print("time consuming:{}".format(end-start),end="  ")
                print("#move: {}".format(move),end="  ")
                print("#sum delta SE: {}".format(delta_sum))

            if move == 0 :
                break

        return self.SE_sort(start)


    def SE_sort(self, start=datetime.now(), verbose=False):
        if verbose:
            print("-" * 30)
            print('Calculate node influence')

        for node, d in self.sorted_nodes:
            s_comm = self.node_comm[node]

            adj_div = {}
            for r_node, w in self.adj[node].items():
                r_comm = self.node_comm[r_node]
                if r_comm != s_comm:
                    if r_comm not in adj_div:
                        adj_div[r_comm] = 0.
                    adj_div[r_comm] += w
            self.node_inf[node] = math.fabs(self.SE_2D(d, self.comm_cut[s_comm], self.comm_vol[s_comm], self.node_deg_in[node], cal_node_inf=True))
            for new_comm, n_in in adj_div.items():
                tar_g = self.comm_cut[new_comm]
                tar_v = self.comm_vol[new_comm]
                self.node_inf[node] += math.fabs(self.SE_2D(d, tar_g, tar_v, n_in, False, cal_node_inf=True))
        if verbose:
            end = datetime.now()
            print("\n")
            print("time consuming:{}".format(end - start))

        return self.mape_node_idx()

    def mape_node_idx(self):
        node_inf_dic={}
        for id, se in enumerate(self.node_inf):
            node_inf_dic[self.nodes[id]] = se

        return node_inf_dic