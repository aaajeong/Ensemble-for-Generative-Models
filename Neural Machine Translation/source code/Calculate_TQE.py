{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "86de0f08",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import numpy as np \n",
    "from tqe import TQE\n",
    "from tqdm import tqdm\n",
    "import re"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b62daa88",
   "metadata": {},
   "source": [
    "### Calculate TQE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9c40e1a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/3 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing embeddings for strings in list 1\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "97c2bbfedea1417da526ebedf75a9240",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing embeddings for strings in list 2\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "de8ceea72a5745f4b0cd78847e3d9d7e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 33%|███▎      | 1/3 [00:02<00:05,  2.93s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing cosine similarity scores\n",
      "Computing embeddings for strings in list 1\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f5e4c71139f44821a5633dccd938cff7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing embeddings for strings in list 2\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9c2a4500c7d04a8aa2d311486f62b101",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 67%|██████▋   | 2/3 [00:05<00:02,  2.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing cosine similarity scores\n",
      "Computing embeddings for strings in list 1\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "57853ebb7cbf405f82c22ecc1e420d6c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing embeddings for strings in list 2\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5a0dad4c981b416cbce449b02736c352",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 3/3 [00:08<00:00,  2.93s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing cosine similarity scores\n",
      "sum:  2.827\n",
      "count:  3\n",
      "mean:  0.9423333333333334\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "f = open('./esb/soft_voting/hpys2.txt', 'r')\n",
    "outputs = f.readlines()\n",
    "\n",
    "f1 = open('./esb/soft_voting/tqe.txt', 'w')\n",
    "sum, count = 0, 0\n",
    "\n",
    "for o in tqdm(outputs):\n",
    "    count +=1\n",
    "    src, hpy, ref = o.split('|')\n",
    "    hpy = hpy.replace('<eos>', '').replace('Result: ', '')\n",
    "    src = src.replace('Source: ', '')\n",
    "    \n",
    "    # source : 번역 원문\n",
    "    # target : 기계 번역 문장\n",
    "    target = []\n",
    "    source = []\n",
    "\n",
    "    # Translation Quality Estimator (QE)\n",
    "    # https://github.com/theRay07/Translation-Quality-Estimator\n",
    "    target.append(hpy)\n",
    "    source.append(src)\n",
    "    model = TQE('LaBSE')\n",
    "    cos_sim_values = model.fit(source, target)\n",
    "    sum += cos_sim_values[0] \n",
    "    \n",
    "    f1.write(str(cos_sim_values[0]))\n",
    "    f1.write('\\n')\n",
    "print('sum: ', sum)\n",
    "print('count: ', count)\n",
    "print('mean: ', sum/count)\n",
    "f1.write('Average TQE: ' + str(sum/count))   \n",
    "\n",
    "f1.close()\n",
    "f.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7.13 ('TQE')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.13"
  },
  "vscode": {
   "interpreter": {
    "hash": "33dd95c08936ee15e8f08616b4cb66f0f57dc9a9815b2c5b868ea546884e803e"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
