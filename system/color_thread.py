import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import torch
from torch import nn, optim


import numpy as np
from queue import Queue
from threading import Thread, Condition, Lock
import random

from time import time, sleep

from network.nslf.hash_1sh import Hash1SH

import asyncio



def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


class EncodeThread():
    def __init__(self, id, device, lr=1e-2, is_train=True, mapping_model='HG', copy_last_mlp=None):
        self.id = id
        self.device=device

        self._requests_cv = Condition()
        self._requests = [False, False]   # requests: [LOCKWINDOW_REQUEST, PROCESS_REQUEST]

        #self._add_data_lock = Lock()

        self._queue = Queue()
        self.memory = []
        self.frame_loss = [] # use to find the worst frame to train

        self.is_train = is_train
        self.left_iters = 10

        self.stopped = False
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.daemon = True # make sure main thread can exit
        self.maintenance_thread.start()
        
        self.status = dict()


        self.xyz=None
        self.rgb=None
        self.direct = None
        self.pred = None

        self.train_count = []


        if mapping_model == 'HG1SH':
            self.model = Hash1SH().to(self.device) # this regress better
            self.opt = optim.Adam(self.model.parameters(), lr=1e-3)# 

        else:
            assert False, 'Not implemented color mapping model [ %s ]'%mapping_model

        self.criterion = torch.nn.MSELoss()

        self.count = 0
        self.trained_iters = 0




    def add_data(self, xyz, direct,rgb,is_train: bool=True):
        if is_train:
            xyz = xyz.cpu().numpy()
            direct = direct.cpu().numpy()
            rgb = rgb.cpu().numpy()
            '''
            if xyz.device != self.device:
                xyz = xyz.cpu().to(self.device)
                direct = derect.cpu().to(self.device)
                rgb = rgb.cpu().to(self.device)
            '''
            xyz = torch.from_numpy(xyz).to(self.device)
            direct = torch.from_numpy(direct).to(self.device)
            rgb = torch.from_numpy(rgb).to(self.device)

        if self.is_train:
            if xyz.shape[0] < 10:
                return
            self._queue.put((xyz,direct,rgb,is_train))
            self.memory.append((xyz,direct,rgb,is_train))
            self.frame_loss.append(1e8)
            #self.train_count.append(1)
        #if not self.is_train:
        else:
            self._queue.put((xyz,direct,rgb,is_train))
            self.left_iters += 1
            with self._requests_cv:
                self._requests_cv.notify()
        print(self.id, len(self.memory))


    def add_iters(self, iters):
        self.left_iters += iters
        with self._requests_cv:
            self._requests_cv.notify()

    def set_iters(self, iters):
        # because feed iter is much much faster than feed data, might block the running
        while(len(self.memory)==0):
            sleep(.01)

        self.left_iters = iters
        if self.left_iters >= 1:
            #print('notify', len(self.memory))
            with self._requests_cv:
                self._requests_cv.notify()
    def close(self):
        self.stopped = True


    def maintenance(self):
        while not self.stopped:
            #with self._requests_cv:
            if not self.is_train or len(self.memory) == 0: # if is train, will load from memory when no feed data
                with self._requests_cv:
                    self._requests_cv.wait()
                #self._requests_cv.acquire()
            if True:
                use_memory = False
                if self._queue.empty():
                    if self.is_train:

                        while len(self.memory)==0:
                            with self._requests_cv:
                                self._requests_cv.wait()

                        if self.left_iters < 1:
                            with self._requests_cv:
                                self._requests_cv.wait()
                        self.left_iters -= 1

                        #frame_counts = np.array(self.train_count)
                        #mem_rand_idx = np.random.choice(np.arange(len(self.train_count)), p=frame_counts/(frame_counts.sum()))
                        mem_rand_idx = np.random.randint(len(self.memory))
                    else: # test
                        with self._requests_cv:
                            self._requests_cv.wait()

                # if has new input, use new input; otherwise with memory
                if not self._queue.empty():# or use_memory:
                    xyz,direct,rgb,is_train = self._queue.get()
                else:#if not use_memory else 
                    xyz,direct,rgb,is_train = self.memory[mem_rand_idx]
                    #self.train_count[mem_rand_idx] += 1
                self._requests[1] = True



                self.xyz = xyz
                self.rgb = rgb
                self.direct = direct

                         
                requests = self._requests[:].copy()
                self._requests[0] = False
                self._requests[1] = False

            self.status['processing'] = True
            if True:#requests[1]:
                st = time()

                if is_train:
                    self.train()
                else:
                    self.eval()
                self.trained_iters += 1
                #print(self.id, 'run_num', self.trained_iters, time()-st, self.xyz.shape[0])
                #print(self.id, 'Encode time ', time()-st)
            self.status['processing'] = False

    def train(self):
        #idx = np.random.choice(self.xyz.shape[0], sample_nm)
        mem_rand_idxs = np.random.randint(len(self.memory),size=min(8, len(self.memory)))
        xyzs,directs,rgbs=[],[],[]
        for mem_rand_idx in mem_rand_idxs:
            xyz,direct,rgb,is_train = self.memory[mem_rand_idx]
            xyzs.append(xyz)
            directs.append(direct)
            rgbs.append(rgb)
        xyz = torch.concat(xyzs,axis=0)
        rgb = torch.concat(rgbs,axis=0)
        direct = torch.concat(directs,axis=0)
        '''
        #mem_rand_idx = np.random.randint(len(self.memory))
        idx = np.argmax(self.frame_loss)
        xyz, direct, rgb, _ = self.memory[idx]
        '''

        with torch.cuda.device(self.device): # NOTE: for multi-gpu, have to use this line!!!!!!!!!!!!!!!!!
            st = time()
            #idx = np.arange(self.xyz.shape[0])
            #pred = self.model(self.xyz[idx,:],self.direct[idx,:])
            pred = self.model(xyz,direct)
            c = pred['c_is'] 
            sigma = pred['sigma_is']

            loss = self.criterion(c, rgb)
            #loss = self.criterion(c, self.rgb[idx,:])

            end = time()
            #print(self.id, loss.item())#, "Num:",c.shape[0], st, end, end-st)
            sys.stdout.flush()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


            self.count += 1
        #self.frame_loss[idx] = (self.frame_loss[idx] + loss.item()) / 2
    def eval(self):
        print(self.id,'Eval')
        self.model.eval()

        pred = self.model(self.xyz,self.direct)
        self.pred = pred['c_is'] 

        self.model.train()
    def eval_w_input(self, xyz, direct):
        print(self.id,'Eval')
        self.model.eval()

        pred = self.model(xyz,direct)
        #self.pred = pred['c_is'] 

        self.model.train()
        return pred['c_is']

