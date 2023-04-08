# REGEN: Recurrent Ensemble Methods for Generative Models*

- Paper Submission 준비중
- 프로젝트 README.md 파일 수정 중




<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> 📖 Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-paper"> ➤ About The Paper</a></li>
    <li><a href="#overview"> ➤ Overview</a></li>
    <li><a href="#project-files-description"> ➤ Project Files Description</a></li>
    <li><a href="#getting-started"> ➤ Getting Started</a></li>
    <li><a href="#references"> ➤ References</a></li>
  </ol>
</details>


<!-- ABOUT THE PAPER -->
<h2 id="about-the-paper"> 📝 About The Paper</h2>

<p align="justify"> 
<ul>
	<li> Published in _ </li>
	<li> Author </li>
	<ol>
	<li><a href="https://github.com/aaajeong">Ahjeong Park</a></li>
	<li><a href="https://github.com/Youngmi-Park">Youngmi Park</a></li>
	<li>Chulyun Kim </li>
	</ol>
</ul>
</p>

<!-- OVERVIEW -->
<h2 id="overview"> 🌞 Overview</h2>

<p align="justify"> 
In this paper, we propose Recurrent Ensemble methods to take advantage of the structural characteristics of recurrent network models. Our proposed algorithms make an agreement at each time step and provide it to all constituent models for the following time step recurrently. To prove effectiveness our method, we experimented various recurrent network models in two tasks(Nerual mahcine Translation and String Arithmetic). Finally we verified that our methods outperform not only single model but also traditional ensembles.</p>

<!-- PROJECT FILES DESCRIPTION -->
<h2 id="project-files-description"> 💾 Project Files Description</h2>
<ol>
<li>Neural Machine Translation</li>
Experiment 1: Translation spainish ➡ english
<ul>
<li><b>nmt_Training.ipynb</b> - Train NMT single models</li>
<li><b>nmt_single.ipynb</b> - Translate single model</li>
<li><b>nmt_esb_softvoting.ipynb</b> - Translate using ensemble with Soft Voting</li>
<li><b>nmt_esb_survival.ipynb</b> - Translate using ensemble with Survival</li>
<li><b>nmt_esb_majority.ipynb</b> - Translate using ensemble with Majority</li>
<li><b>calcualte_TQE.ipynb</b> - Evaluate Translation Quality for all ensemble models and single model</li>
<li><b>get_graph.ipynb</b> - Graph for results</li>
</ul>
<li>String Arithmetic</li>
Experiments 2: Arithmetic as String
<ul>
<li><b>train_single.py</b> - Train/Evaluate String Arithmetic single model</li>
<li><b>train_esb_softvoting.py</b> - Train/Evaluate String Arithmetic ensemble with Soft Voting</li>
<li><b>train_esb_survival.py</b> - Train/Evaluate String Arithmetic ensemble with Survival</li>
<li><b>train_esb_majority.py</b> - Train/Evaluate String Arithmetic ensemble with Majority</li>
<li><b>get_graph.py</b> - Graph for results</li>
</ul>
</ol>


<h3>Some other supporting files</h3>
<ol>
<li>Neural Machine Translation</li>
<ul>
<li><b>dataset</b> - Using <a href = "http://www.manythings.org/anki/">Spanish - English dataset</a> with random shuffle.</li>
<li><b>TQE(5-Epoch).xlsx</b> - All Results for our experiments(source, target sentence and TQE per 2 epoch)</li>
<li><b>Checkpoint</b> - Training takes too much time. You can download checkpoint in <a href = "https://drive.google.com/drive/folders/1JyyEol3SmREQok6fJ5RqrtkRUPRMwMry?usp=sharing">here</a>.</li>
</ul>
<li>String Arithmetic</li>
<ul>
<li><b>dataset</b> - string arithmetic dataset consisted of 3 digit & 1 operator</li>
<li><b>common</b> - base model, functions, layers, trainer etc.</li>
<li><b>training_memo</b> - Training memo per epoch for all models(single, ensemble with soft voting, survival and majority)</li>
</ul>
</ol>

<!-- GETTING STARTED -->
<h2 id="getting-started"> 📖 Getting Started</h2>
<p>To clone and run this code, you'll need <a href = "https://git-scm.com">Git</a> installed on your computer. From your command line:</p>

    # Clone this repository
    $ git clone https://github.com/aaajeong/Recurrent-Ensemble.git
    
    # Go into the repository
    $ cd Recurrent-Ensemble
    
    # Create Environment
    $ conda env create -f recur_esb.yaml
    $ conda activate recur_esb
    
    
<ol>
<li>Neural Machine Translation</li>
<pre><code>You are able to train/test code by running jupyter notebook.</code></pre>
<pre><code>You can also evaluate TQE by running jupyter notebook.
PyTorch 1.2.0 or higher is recommended. If the install below gives an error, please install pytorch first. </code></pre>
<li>String Arithmetic</li>
<p>You can train/test the code by running like this:
<pre><code>$ python train_single.py</code></pre>
</ol>

<!-- REFERENCES -->
<h2 id="references"> 📖 References</h2>
<i>Yasuki Saito, Deep Learning from scratch 2, hanbit(July 2018), chapter 07</i>
<br>
<i>Tensorflow Tutorials: <a href = 
"https://github.com/tensorflow/text/blob/master/docs/tutorials/nmt_with_attention.ipynb">Neural mahcine translation with Attention</a></i>
